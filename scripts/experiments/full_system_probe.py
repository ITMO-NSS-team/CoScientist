#!/usr/bin/env python
"""Full-system REAL probe: bridge A vs B on real dataset_S / dataset_L tasks.

NOTHING is stubbed here — real grounding (RAG/Postgres), real sub-agents
(TaskExecutorAgent -> FEDOT.MAS -> MCP tools -> S3), real critic. This exercises the
whole pipeline with the planner+critic we built:

  * submit_plan FunctionTool (R09) — orchestrator must submit a plan first; the handler
    validates it + runs the deterministic gate (R12) against the LIVE inventory it actually
    retrieved this session (state['seen_inventory'], accumulated by the wrapped grounding
    tools), with the fidelity fix (exact schema errors + schema/example in the description).
  * delegation-gate critic (advisory; never terminates) on the first delegation per agent.
  * bridge_A (plan-as-contract): PlanReAct drives real delegations; the callback checks
    each against the plan (conformance) and nudges off-plan ones.
  * bridge_B (DAG-executor): on an accepted plan, submit_plan IMMEDIATELY executes the DAG
    over the REAL sub-agents (AgentTool.run_async in topological order, {artifact_id}
    substituted); the orchestrator then just finalizes.

Each run: >=30s pause AFTER it (the user's requirement — MCP tools are not async, don't
overload them between tasks). Every run's params+outcome are appended to a manifest JSONL
immediately, so a crash mid-battery never loses prior results. session_id == Opik thread_id.

Usage:
  python scripts/experiments/full_system_probe.py --feasibility           # 1 A + 1 B on GSK only
  python scripts/experiments/full_system_probe.py --battery --cap 1600     # full 10 + 1 no-fix
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("LLM__MAIN_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
os.environ.setdefault("HITL__HEADLESS_AUTO_APPROVE", "true")
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/opik_eval"))

from google.adk.tools import FunctionTool                       # noqa: E402
from google.genai import types as genai_types                   # noqa: E402

import CoScientist.agents.agents as A                           # noqa: E402
from CoScientist.agents.agents import ResilientAgentTool        # noqa: E402
from CoScientist.agents import catalog                          # noqa: E402
from CoScientist.agents.critic_agent import (                   # noqa: E402
    _extract_pending_calls, _invoke_critic_llm, _session_contents,
    _extract_completed_trajectory, _format_trajectory, _format_pending_calls,
    _first_text, _apply_revisions, PreVerdict,
)
from CoScientist.agents.prompts import pre_action_critic_instruction  # noqa: E402
from CoScientist.experiments.plan import ExperimentPlan         # noqa: E402
from CoScientist.experiments.executor import execute_plan       # noqa: E402
from CoScientist.experiments.bridge import check_conformance    # noqa: E402
from CoScientist.experiments.submit_plan_tool import run_submit_plan, _RICH_DOC, _TERSE_DOC  # noqa: E402
import CoScientist.tools.retrieval_tools as RT                  # noqa: E402
from CoScientist.agents.llm_repair import install_json_repair  # noqa: E402

try:
    from trace_locator import record_run                        # noqa: E402
except Exception:
    record_run = None

ROSTER = {s.name for s in catalog.enabled_agents()}

# Fix #1 (F015a.A4): repair qwen's malformed tool-call JSON at the LiteLlm boundary
# so a JSONDecodeError no longer kills the whole run.
install_json_repair()


class TimeoutResilientAgentTool(ResilientAgentTool):
    """Fix #2/#3: bound each delegation and convert an MCP/tool timeout into a NON-FATAL
    marker, so the orchestrator continues and the finalizer can still deliver partial
    results (a 300s McpError no longer sinks the whole run)."""

    PER_CALL_TIMEOUT = 700

    async def run_async(self, *, args, tool_context):
        try:
            return await asyncio.wait_for(
                super().run_async(args=args, tool_context=tool_context),
                timeout=self.PER_CALL_TIMEOUT)
        except asyncio.TimeoutError:
            return (f"[{self.agent.name}] step exceeded {self.PER_CALL_TIMEOUT}s and was cut off; "
                    "moving on — any results produced before the cutoff are saved to S3/state.")
        except Exception as e:
            n = type(e).__name__
            if "Mcp" in n or "Timed out" in str(e) or "timeout" in str(e).lower():
                return (f"[{self.agent.name}] an MCP/tool call timed out ({n}); moving on — "
                        "partial results (if any) are saved to S3/state.")
            raise


# ── wrapped REAL grounding — accumulate the live inventory the orchestrator saw ──
async def list_available_tools(query: str, tool_context=None) -> dict:
    """Real RAG retrieval, but record (server, tool) seen so the gate can check resolvability."""
    res = await RT.list_available_tools(query)
    if tool_context is not None and res.get("tools"):
        seen = tool_context.state.get("seen_inventory", {})
        for t in res["tools"]:
            srv = t.get("server_id") or "UNKNOWN"
            lst = set(seen.get(srv, []))
            lst.add(t.get("name"))
            seen[srv] = sorted(x for x in lst if x)
        tool_context.state["seen_inventory"] = seen
    return res


async def list_server_tools(server_id: str, tool_context=None) -> dict:
    res = await RT.list_server_tools(server_id)
    if tool_context is not None and res.get("tools"):
        seen = tool_context.state.get("seen_inventory", {})
        lst = set(seen.get(server_id, []))
        for t in res["tools"]:
            if t.get("name"):
                lst.add(t["name"])
        seen[server_id] = sorted(lst)
        tool_context.state["seen_inventory"] = seen
    return res


def _agent_tools_map() -> dict:
    return {s.name: TimeoutResilientAgentTool(agent=A._resolve_agent(s.name)) for s in catalog.enabled_agents()}


def _make_submit_plan(bridge: str, fidelity_fix: bool, agent_tools: dict) -> FunctionTool:
    async def submit_plan(plan: dict, tool_context=None) -> dict:
        state = tool_context.state if tool_context is not None else None
        inv = (state or {}).get("seen_inventory", {})
        res = await run_submit_plan(plan, inv, state=state, verbose_errors=fidelity_fix)
        if state is not None:
            log = state.get("submit_plan_calls", [])
            log.append({"accepted": res["accepted"], "gate_code": res["gate_code"],
                        "n_steps": res.get("n_steps"), "order": res.get("order")})
            state["submit_plan_calls"] = log
        if bridge == "bridge_B" and res["accepted"] and state is not None:
            # DAG-executor over the REAL sub-agents.
            p = ExperimentPlan(**state["experiment_plan"])

            async def dispatch(step, agent_name, params):
                tool = agent_tools.get(agent_name)
                if tool is None:
                    return f"[no agent for kind={step.kind}]"
                req = f"{step.subtask}\nParameters: {json.dumps(params, ensure_ascii=False)}"
                return await tool.run_async(args={"request": req}, tool_context=tool_context)

            ex = await execute_plan(p, dispatch)
            state["executor_result"] = {"completed": ex["completed"],
                                        "steps_run": [t["step"] for t in ex["trace"]],
                                        "n_steps": len(ex["trace"])}
            summary = "; ".join(f"{t['step']}({t['agent']}):{'ok' if t['ok'] else 'FAIL'}" for t in ex["trace"])
            tail = "\n".join(f"[{t['step']}] {str(t['result'])[:600]}" for t in ex["trace"])
            return {**res, "executed": True, "completed": ex["completed"],
                    "result": f"Executed plan in order: {summary}.\nStep results:\n{tail}\nYou may finalize."}
        return res

    submit_plan.__doc__ = _RICH_DOC if fidelity_fix else _TERSE_DOC
    return FunctionTool(submit_plan)


def _make_callback(bridge: str):
    """after_model_callback: turn counter + advisory delegation-gate critic + (A) conformance."""
    async def cb(callback_context, llm_response):
        pending = _extract_pending_calls(llm_response)
        if not pending:
            return None
        st = callback_context.state
        st["_turns_with_calls"] = st.get("_turns_with_calls", 0) + 1
        deleg = [c for c in pending if c["tool"] in ROSTER]
        if not deleg:
            return None

        # Bridge A: deterministic conformance nudge (no LLM).
        if bridge == "bridge_A" and st.get("experiment_plan"):
            plan = ExperimentPlan(**st["experiment_plan"])
            done = st.get("_conformance_done_ids", [])
            clog = st.get("conformance_log", [])
            for c in deleg:
                ok, reason = check_conformance(c["tool"], plan, done)
                clog.append({"agent": c["tool"], "ok": ok, "reason": reason})
                if not ok and llm_response.content and llm_response.content.parts:
                    llm_response.content.parts.insert(
                        0, genai_types.Part(text=f"[PLAN-CONFORMANCE]: {reason}. Follow the submitted plan.",
                                            thought=True))
            st["conformance_log"] = clog

        # Advisory delegation-gate critic: first delegation per new agent (never terminates).
        critiqued = set(st.get("_critiqued_agents", []))
        new_targets = [c for c in deleg if c["tool"] not in critiqued]
        if not new_targets:
            return None
        st["_critiqued_agents"] = sorted(critiqued | {c["tool"] for c in new_targets})
        contents = _session_contents(callback_context)
        user_task = _first_text(getattr(callback_context, "user_content", None))
        traj = _extract_completed_trajectory(contents)
        up = (f"ORIGINAL TASK:\n{user_task}\n\nCOMPLETED TRAJECTORY:\n{_format_trajectory(traj)}\n\n"
              f"PROPOSED NEXT ACTION(S):\n{_format_pending_calls(pending)}\n\n"
              "Approve, revise, or reject. Respond as strict JSON.")
        payload = await _invoke_critic_llm(pre_action_critic_instruction, up)
        verdict = (payload.get("verdict") or "approve").lower().strip()
        fb = (payload.get("feedback") or "").strip()
        h = st.get("critic_pre_history", [])
        h.append({"verdict": verdict, "feedback": fb,
                  "proposed": [{"tool": c["tool"]} for c in pending]})
        st["critic_pre_history"] = h
        if verdict == PreVerdict.REVISE.value and (payload.get("revised_calls") or []):
            _apply_revisions(pending, payload["revised_calls"])
        if verdict == PreVerdict.REJECT.value and llm_response.content and llm_response.content.parts:
            llm_response.content.parts.insert(0, genai_types.Part(
                text=f"[CRITIC NOTE — reconsider]: {fb}", thought=True))
        return None
    return cb


_PLAN_NUDGE = ("\n\n### PLAN-FIRST (REQUIRED)\nBefore delegating to ANY sub-agent, call "
               "`submit_plan(plan=...)` ONCE with your full plan as a JSON DAG (each step: id, "
               "subtask, kind, tool_servers taken from list_available_tools results, deps). Only "
               "delegate after submit_plan returns accepted=true.")


def install(bridge: str, fidelity_fix: bool) -> None:
    agent_tools = _agent_tools_map()
    A.orchestrator_agent.tools = [
        _make_submit_plan(bridge, fidelity_fix, agent_tools),
        FunctionTool(list_available_tools),
        FunctionTool(list_server_tools),
        *agent_tools.values(),
    ]
    A.orchestrator_agent.after_model_callback = _make_callback(bridge)
    base = A.orchestrator_agent.instruction
    if "PLAN-FIRST (REQUIRED)" not in (base or ""):
        A.orchestrator_agent.instruction = (base or "") + _PLAN_NUDGE


# ── tasks ──
S_TASKS = [
    ("S_gsk", "Generate GSK-3beta inhibitors with high activity"),
    ("S_kras_sel", "Generate inhibitors of KRAS protein with G12C mutation. The inhibitors should be "
                   "selective, meaning they should not bind with HRAS and NRAS proteins."),
    ("S_stat3", "Can you suggest molecules that inhibit signal transducer and activator of transcription 3 "
                "(STAT3) with water solubility greater than 60 ug/mL and inhibitory ability to P450 CYP1A2?"),
]


def load_L_tasks(n: int = 2) -> list:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(REPO / "CoScientist/dataset_L.xlsx", read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        hdr = [str(h) for h in rows[0]]
        out, seen = [], set()
        for r in rows[1:]:
            d = dict(zip(hdr, r))
            q = str(d.get("content") or "").strip()
            if not q or q in seen:
                continue
            seen.add(q)
            out.append((f"L_{len(out)+1}", q))
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print(f"[warn] could not load dataset_L: {e}", flush=True)
        return []


async def run_one(spec: dict, cap: int, manifest_path: pathlib.Path) -> dict:
    from CoScientist.main import CoScientistManager
    bridge, fidelity_fix = spec["bridge"], spec["fidelity_fix"]
    sid = f"fs_{bridge}_{spec['task_id']}_{'fix' if fidelity_fix else 'nofix'}_{spec['stamp']}"
    install(bridge, fidelity_fix)
    mgr = CoScientistManager(session_id=sid)
    t0 = time.time(); err, resp = None, ""
    try:
        resp = await asyncio.wait_for(mgr.run(spec["task"], verbose=False), timeout=cap)
    except asyncio.TimeoutError:
        err = f"timeout>{cap}s"
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:200]}"
    st = {}
    try:
        sess = await mgr.session_service.get_session(app_name=mgr.app_name, user_id=mgr.user_id, session_id=sid)
        st = dict(getattr(sess, "state", {}) or {})
    except Exception:
        pass
    try:
        await mgr.close()
    except Exception:
        pass
    sp = st.get("submit_plan_calls", [])
    acc = [c for c in sp if c["accepted"]]
    conf = st.get("conformance_log", [])
    ex = st.get("executor_result", {})
    arts = st.get("fedot_artifacts") or []
    artifact_urls = [a.get("url") for a in arts if isinstance(a, dict) and a.get("url")]
    rec = {
        "bridge": bridge, "task_id": spec["task_id"], "dataset": spec["task_id"][0],
        "fidelity_fix": fidelity_fix, "cap": cap, "session_id": sid,
        "duration_s": round(time.time() - t0, 1), "error": err, "resp_len": len(resp or ""),
        "resp_head": (resp or "")[:300], "resp_tail": (resp or "")[-1200:],
        "artifact_urls": artifact_urls,
        "has_s3": ("X-Amz-Signature" in (resp or "")) or ("10.32.1.114:9000" in (resp or "")),
        "submit_plan_attempts": len(sp), "submit_plan_accepted": bool(acc),
        "plan_fill_fidelity": (len(acc) / len(sp)) if sp else 0.0,
        "first_gate_code": sp[0]["gate_code"] if sp else "not_called",
        "accepted_order": acc[0]["order"] if acc else None,
        "turns": st.get("_turns_with_calls", 0),
        "critic_firings": len(st.get("critic_pre_history", [])),
        "conformance_checks": len(conf), "off_plan": sum(1 for c in conf if not c["ok"]),
        "executor_completed": ex.get("completed"), "executor_steps": ex.get("n_steps"),
        "seen_servers": sorted((st.get("seen_inventory", {}) or {}).keys()),
    }
    # trace id (best-effort)
    rec["trace_id"] = None
    if record_run:
        try:
            entry = await asyncio.wait_for(asyncio.to_thread(
                record_run, sid, query=spec["task"], condition=bridge,
                model=os.environ["LLM__MAIN_MODEL"]), timeout=60)
            rec["trace_id"] = entry["trace_id"] if entry else None
        except Exception:
            pass
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility", action="store_true", help="1 bridge_A + 1 bridge_B on GSK only")
    ap.add_argument("--battery", action="store_true", help="3 S + 2 L x 2 bridges + 1 no-fix")
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--delay", type=int, default=30, help="seconds to wait AFTER each run")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%H%M%S")
    L_TASKS = load_L_tasks(2)
    tasks = S_TASKS + L_TASKS

    specs = []
    if args.feasibility:
        specs = [{"bridge": "bridge_A", "task_id": "S_gsk", "task": S_TASKS[0][1], "fidelity_fix": True},
                 {"bridge": "bridge_B", "task_id": "S_gsk", "task": S_TASKS[0][1], "fidelity_fix": True}]
    else:  # battery
        for b in ("bridge_A", "bridge_B"):
            for tid, txt in tasks:
                specs.append({"bridge": b, "task_id": tid, "task": txt, "fidelity_fix": True})
        # one no-fix comparison run (bridge_B on GSK)
        specs.append({"bridge": "bridge_B", "task_id": "S_gsk", "task": S_TASKS[0][1], "fidelity_fix": False})
    for s in specs:
        s["stamp"] = stamp

    outdir = REPO / "scripts/experiments/results"; outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / f"full_system_{datetime.date.today().isoformat()}_{stamp}.jsonl"
    print(f"=== full-system probe · {len(specs)} runs · cap={args.cap}s · delay={args.delay}s · "
          f"manifest={manifest.name} ===", flush=True)
    for i, spec in enumerate(specs):
        print(f"[{i+1}/{len(specs)}] {spec['bridge']} {spec['task_id']} "
              f"fix={spec['fidelity_fix']}", flush=True)
        rec = await run_one(spec, args.cap, manifest)
        print(f"    -> {rec['duration_s']}s len={rec['resp_len']} s3={rec['has_s3']} "
              f"submit_plan(acc={rec['submit_plan_accepted']},gate={rec['first_gate_code']},"
              f"order={rec['accepted_order']},fidelity={rec['plan_fill_fidelity']:.2f}) "
              f"turns={rec['turns']} off_plan={rec['off_plan']}/{rec['conformance_checks']} "
              f"exec_done={rec['executor_completed']}/{rec['executor_steps']} "
              f"trace={rec['trace_id']} err={rec['error']}", flush=True)
        if i < len(specs) - 1:
            print(f"    (waiting {args.delay}s before next run — MCP cool-down)", flush=True)
            await asyncio.sleep(args.delay)
    print(f"\n=== done · manifest {manifest} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
