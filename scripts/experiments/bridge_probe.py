#!/usr/bin/env python
"""Integration probe comparing the two plan->delegation bridges with the LIVE orchestrator.

Both bridges share: the orchestrator is prompted to call submit_plan(plan) FIRST (a real
FunctionTool that validates the ExperimentPlan + runs the deterministic gate). They differ
in WHAT happens after a plan is accepted:

  bridge_A (plan-as-contract): submit_plan only validates+stores state['experiment_plan'];
      PlanReAct keeps driving; the after_model_callback checks each delegation for
      CONFORMANCE to the plan (on-plan vs off-plan) and nudges off-plan ones.

  bridge_B (DAG-executor): submit_plan, on accept, IMMEDIATELY runs execute_plan() over the
      DAG (dispatching each step to the stubbed sub-agents in topological order, substituting
      {artifact_id}) and returns the executed trace; the orchestrator then just finalizes.

FEDOT-free (sub-agents stubbed, grounding = frozen inventory), so only orchestrator+critic
are live qwen. Metrics read offline from session state. Clean stub (sha256 signature, no
'stub'/'fabricated' tells) per the wave-2 honesty guards.

Usage:
  python scripts/experiments/bridge_probe.py --smoke
  python scripts/experiments/bridge_probe.py --bridges bridge_A,bridge_B --probes ctrl,kras,gap --repeats 2 --cap 240
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
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
from google.adk.tools.agent_tool import AgentTool               # noqa: E402

import CoScientist.agents.agents as A                           # noqa: E402
from CoScientist.agents import catalog                          # noqa: E402
from CoScientist.agents.critic_agent import _extract_pending_calls  # noqa: E402
from CoScientist.experiments.plan import ExperimentPlan         # noqa: E402
from CoScientist.experiments.gate import deterministic_gate     # noqa: E402
from CoScientist.experiments.executor import execute_plan       # noqa: E402
from CoScientist.experiments.bridge import check_conformance    # noqa: E402
from CoScientist.experiments.submit_plan_tool import run_submit_plan  # noqa: E402

ROSTER = {s.name for s in catalog.enabled_agents()}

# ── frozen inventory (same stand-in as plan_critic_probe / r05) ──
INVENTORY = {
    "generative-models-mcp": [
        ("generate_case_mols", "generate candidate molecules for a known target case "
         "(cases: alzheimer, parkinson, cancer, sclerosis, dyslipidemia, drug_resistance)"),
        ("generate_mols", "sample molecules from the generic generative model"),
    ],
    "chemical-mcp-server": [
        ("calculate_docking", "docking score of a SMILES against a protein (pdb id)"),
        ("get_rdkit_properties", "RDKit physchem properties (MW, logP, TPSA, QED)"),
    ],
    "admet-mcp": [("predict_admet", "ADMET / toxicity / solubility / CYP endpoints from SMILES")],
}

# clean presigned URL — realistic 64-hex signature, no "stub" tell
_SIG = hashlib.sha256(b"coscientist-mols-20260614").hexdigest()
_PRESIGNED = (f"https://storage.itmo.ru/coscientist/mols_20260614.csv?"
              f"X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600&X-Amz-Signature={_SIG}")
_CANNED = {
    "TaskExecutorAgent": (
        f"Generated 12 candidate molecules via generate_case_mols. Results at S3: {_PRESIGNED} "
        "| columns: Smiles, QED, docking_score | generated_count: 12."),
    "ResearchAgent": ("GSK-3beta is an established Alzheimer's target; in the generative tooling the "
                      "'alzheimer' case corresponds to GSK-3beta."),
    "CoderAgent": ("Downloaded the CSV from the presigned URL and parsed 12 rows. Sample: "
                   "O=C1Cc2ccccc2N1C(=O)c1ccc(F)cc1 (QED 0.81, docking -9.2). All rows extracted."),
    "HypothesesAgent": "Approach: generate candidates for the target case, score by docking + ADMET, filter.",
    "MedicalAgent": "Clinical decision-support summary for the query.",
}


# ── grounding stubs (frozen inventory) ──
async def list_available_tools(query: str) -> dict:
    tools = [{"name": n, "server_id": srv, "description": d, "score": 0.9}
             for srv, items in INVENTORY.items() for n, d in items]
    return {"status": "success", "count": len(tools), "tools": tools}


async def list_server_tools(server_id: str) -> dict:
    items = INVENTORY.get(server_id)
    if items is None:
        return {"status": "unavailable", "server_id": server_id, "count": 0, "tools": []}
    return {"status": "success", "server_id": server_id, "count": len(items),
            "tools": [{"name": n, "description": d, "input_schema": {}, "tags": []} for n, d in items]}


class StubAgentTool(AgentTool):
    async def run_async(self, *, args, tool_context):
        if isinstance(args, dict) and "request" not in args:
            alt = next((v for v in args.values() if isinstance(v, str) and v.strip()), None)
            args = {**args, "request": alt or "Proceed."}
        return _CANNED.get(self.agent.name, f"[{self.agent.name}] done")


# ── inventory in the {server: [tool]} shape the gate wants ──
_INV_NAMES = {srv: [n for n, _ in items] for srv, items in INVENTORY.items()}


async def _stub_dispatch(step, agent, params):
    """Bridge B dispatch: return the canned result for the step's mapped agent."""
    return _CANNED.get(agent, f"[{agent}] done")


def _make_submit_plan(bridge: str) -> FunctionTool:
    """submit_plan tool. For bridge_B it also RUNS the executor on accept."""
    async def submit_plan(plan: dict, tool_context=None) -> dict:
        """Submit the full experiment plan (JSON DAG of steps) for validation BEFORE any delegation.
        Each step: {id, subtask, kind: compute|research|hypothesize|code_exec,
        tool_servers:[{server,tools}], run_params, expected_artifacts:[{id,description}], deps}.
        A 'compute' step MUST name >=1 tool_server from the available tools. Returns
        {accepted, gate_code, detail, order}; if not accepted, fix per gate_code and resubmit."""
        state = tool_context.state if tool_context is not None else None
        res = await run_submit_plan(plan, _INV_NAMES, state=state)
        # log every attempt for offline metrics
        if state is not None:
            log = state.get("submit_plan_calls", [])
            log.append({"accepted": res["accepted"], "gate_code": res["gate_code"],
                        "n_steps": res.get("n_steps"), "order": res.get("order")})
            state["submit_plan_calls"] = log
        if bridge == "bridge_B" and res["accepted"] and state is not None:
            # DAG-executor takes over: run the whole plan deterministically, return the trace.
            p = ExperimentPlan(**state["experiment_plan"])
            ex = await execute_plan(p, _stub_dispatch)
            state["executor_result"] = {"completed": ex["completed"],
                                        "steps_run": [t["step"] for t in ex["trace"]]}
            steps_summary = "; ".join(f"{t['step']}({t['agent']})->done" for t in ex["trace"])
            return {**res, "executed": True, "completed": ex["completed"],
                    "result": f"Plan executed in order: {steps_summary}. Final artifacts ready. "
                              f"Generated molecules at {_PRESIGNED}. You may finalize."}
        return res
    return FunctionTool(submit_plan)


def _make_callback(bridge: str):
    """after_model_callback: counts turns; for bridge_A checks delegation conformance to the plan."""
    async def cb(callback_context, llm_response):
        pending = _extract_pending_calls(llm_response)
        if not pending:
            return None
        st = callback_context.state
        st["_turns_with_calls"] = st.get("_turns_with_calls", 0) + 1
        if bridge != "bridge_A":
            return None
        plan_d = st.get("experiment_plan")
        deleg = [c for c in pending if c["tool"] in ROSTER]
        if not plan_d or not deleg:
            return None
        plan = ExperimentPlan(**plan_d)
        done = st.get("_conformance_done_ids", [])
        log = st.get("conformance_log", [])
        for c in deleg:
            ok, reason = check_conformance(c["tool"], plan, done)
            log.append({"agent": c["tool"], "ok": ok, "reason": reason})
            if not ok and llm_response.content and llm_response.content.parts:
                llm_response.content.parts.insert(
                    0, __import__("google.genai.types", fromlist=["Part"]).Part(
                        text=f"[PLAN-CONFORMANCE]: {reason}. Follow the submitted plan.", thought=True))
        st["conformance_log"] = log
        return None
    return cb


def install(bridge: str) -> None:
    stub_subagents = [StubAgentTool(agent=A._resolve_agent(s.name)) for s in catalog.enabled_agents()]
    A.orchestrator_agent.tools = [
        _make_submit_plan(bridge),
        FunctionTool(list_available_tools),
        FunctionTool(list_server_tools),
        *stub_subagents,
    ]
    A.orchestrator_agent.after_model_callback = _make_callback(bridge)
    # prompt nudge: plan-first
    base = A.orchestrator_agent.instruction
    nudge = ("\n\n### PLAN-FIRST (REQUIRED)\nBefore delegating to ANY sub-agent, call "
             "`submit_plan(plan=...)` ONCE with your full plan as a JSON DAG of steps "
             "(each step: id, subtask, kind, tool_servers from the available tools, deps). "
             "Only delegate after submit_plan returns accepted=true.")
    if "PLAN-FIRST (REQUIRED)" not in (base or ""):
        A.orchestrator_agent.instruction = (base or "") + nudge


PROBES = {
    "ctrl": "Generate GSK-3beta inhibitors with high activity",
    "kras": ("Generate inhibitors of KRAS protein with G12C mutation. They should be selective "
             "(not bind HRAS and NRAS)."),
    "gap": ("Generate KRAS G12C inhibitors, then run a full clinical-trial simulation with Phase II "
            "success probability"),
}
BRIDGES = ["bridge_A", "bridge_B"]


async def run_one(bridge: str, probe: str, rep: int, cap: int, stamp: str) -> dict:
    from CoScientist.main import CoScientistManager
    sid = f"br_{bridge}_{probe}_{rep}_{stamp}"
    install(bridge)
    mgr = CoScientistManager(session_id=sid)
    t0 = time.time(); err, resp = None, ""
    try:
        resp = await asyncio.wait_for(mgr.run(PROBES[probe], verbose=False), timeout=cap)
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
    accepted = [c for c in sp if c["accepted"]]
    conf = st.get("conformance_log", [])
    ex = st.get("executor_result", {})
    return {
        "bridge": bridge, "probe": probe, "rep": rep, "session_id": sid,
        "duration_s": round(time.time() - t0, 1), "error": err, "resp_len": len(resp or ""),
        "submit_plan_attempts": len(sp),
        "submit_plan_accepted": bool(accepted),
        "plan_fill_fidelity": (len(accepted) / len(sp)) if sp else 0.0,
        "first_gate_code": sp[0]["gate_code"] if sp else "not_called",
        "accepted_order": accepted[0]["order"] if accepted else None,
        "turns": st.get("_turns_with_calls", 0),
        # bridge A:
        "conformance_checks": len(conf),
        "off_plan": sum(1 for c in conf if not c["ok"]),
        # bridge B:
        "executor_completed": ex.get("completed"),
        "executor_steps": ex.get("steps_run"),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridges", default=",".join(BRIDGES))
    ap.add_argument("--probes", default=",".join(PROBES))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cap", type=int, default=240)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        bridges, probes, reps = ["bridge_A", "bridge_B"], ["ctrl"], 1
    else:
        bridges = [b.strip() for b in args.bridges.split(",") if b.strip()]
        probes = [p.strip() for p in args.probes.split(",") if p.strip()]
        reps = args.repeats
    stamp = datetime.datetime.now().strftime("%H%M%S")
    total = len(bridges) * len(probes) * reps
    print(f"=== bridge probe · {bridges} × {probes} × {reps} = {total} runs · cap={args.cap}s ===", flush=True)
    runs = []
    i = 0
    for b in bridges:
        for p in probes:
            for r in range(reps):
                i += 1
                print(f"[{i}/{total}] {b} {p}", flush=True)
                rec = await run_one(b, p, r, args.cap, stamp)
                print(f"    -> {rec['duration_s']}s submit_plan(acc={rec['submit_plan_accepted']},"
                      f"gate={rec['first_gate_code']},order={rec['accepted_order']}) turns={rec['turns']} "
                      f"off_plan={rec['off_plan']}/{rec['conformance_checks']} "
                      f"exec_done={rec['executor_completed']} steps={rec['executor_steps']} err={rec['error']}",
                      flush=True)
                runs.append(rec)
    outdir = REPO / "scripts/experiments/results"; outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"bridge_{datetime.date.today().isoformat()}_{stamp}.json"
    out.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1))
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
