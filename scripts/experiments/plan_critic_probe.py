#!/usr/bin/env python
"""FEDOT-free plan-critic probe — compare critic MODES on plan detection/quality.

Runs the REAL OrchestratorAgent (live qwen + PlanReActPlanner + after_model_callback)
but with EVERYTHING expensive stubbed, so a run is seconds and deterministic:

  * sub-agents (TaskExecutorAgent / ResearchAgent / CoderAgent / Hypotheses / Medical)
    are replaced by StubAgentTool — a real AgentTool subclass (byte-identical name +
    description, so function_call.name routing is prod-identical) that returns a canned,
    realistic result instantly and seeds state['fedot_artifacts'] like the real
    fedot_artifact_plugin would. NO FEDOT.MAS, NO MCP, NO docking, NO network.
  * the orchestrator's OWN grounding tools list_available_tools / list_server_tools are
    backed by a FROZEN real MCP inventory (reused from r05_plan_benchmark) instead of the
    live Postgres/Qdrant RAG — so the orchestrator STILL analyses its tools and plans on
    them (the user's requirement), but with no VPN/DB dependency.

The only live LLM calls are the orchestrator's own turns and the critic's — i.e. exactly
the thing under test. Headline metrics come from the critic's own audit trail in session
state (state['critic_pre_history']) read offline after each run — no Opik needed for the
result; Opik is recorded as a cross-check.

CONDITIONS (critic modes), all on the current PlanReAct substrate:
  none         after_model_callback = None
  per-action   pre_action_critique, plan_critic_only=False   (current default; churn + FP)
  tags         pre_action_critique, plan_critic_only=True    (current plan-critic; tag scan)
  delegation   delegation_gate_critique (Design 1)           (fire on first roster delegation)

HONEST SCOPE (from the adversarial confound review): stubbing faithfully isolates
DETECTION (fire-rate) and FALSE-POSITIVE (alzheimer) — those depend on the real-qwen
plan-forming turn + byte-identical AgentTool names, so they transfer to prod. It does
NOT isolate replan-rate / #critic-calls / latency to prod MAGNITUDES (prod's
result-driven replan path is removed and post_action_critique is disabled at
agents.py:321) — those are valid only as WITHIN-stub relative comparisons. Reported as such.

Usage:
  python scripts/experiments/plan_critic_probe.py --smoke
  python scripts/experiments/plan_critic_probe.py --conditions none,per-action,tags,delegation \
      --probes ctrl,fp_alz,gap,wrong --repeats 2 --cap 200
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

# Force qwen + headless auto-approve BEFORE importing CoScientist (the agents/tracer
# read the model at import, exactly like ab_runner.py).
os.environ.setdefault("LLM__MAIN_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
os.environ.setdefault("HITL__HEADLESS_AUTO_APPROVE", "true")

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/opik_eval"))

from google.adk.tools import FunctionTool                          # noqa: E402
from google.adk.tools.agent_tool import AgentTool                  # noqa: E402

import CoScientist.agents.agents as A                              # noqa: E402
from CoScientist.agents import catalog                             # noqa: E402
from CoScientist.config.settings import get_settings               # noqa: E402
from CoScientist.agents.critic_agent import (                      # noqa: E402
    pre_action_critique,
    _extract_pending_calls,
    _invoke_critic_llm,
    _session_contents,
    _extract_completed_trajectory,
    _format_trajectory,
    _format_pending_calls,
    _first_text,
    _apply_revisions,
    PreVerdict,
)
from CoScientist.agents.prompts import pre_action_critic_instruction  # noqa: E402
from google.genai import types                                    # noqa: E402

try:
    from trace_locator import record_run                          # noqa: E402
except Exception:
    record_run = None

ROSTER = {s.name for s in catalog.enabled_agents()}

# ─────────────────────────────────────────────────────────────────────────────
# Frozen MCP inventory (reused from r05_plan_benchmark.py — a stand-in for the live
# tool index). {server: [(tool, desc), ...]}. Drives the stubbed grounding tools.
# ─────────────────────────────────────────────────────────────────────────────
INVENTORY = {
    "generative-models-mcp": [
        ("generate_case_mols", "generate candidate molecules for a known target case "
         "(supported cases: alzheimer, parkinson, cancer, sclerosis, dyslipidemia, drug_resistance)"),
        ("generate_mols", "sample molecules from the generic trained generative model (no case needed)"),
        ("list_generative_train_cases", "list the disease cases the generator supports"),
    ],
    "chemical-mcp-server": [
        ("calculate_docking", "docking score of a SMILES against a protein (pdb id)"),
        ("get_rdkit_properties", "RDKit physchem properties from SMILES (MW, logP, TPSA, QED)"),
        ("retrosynthesis_route", "retrosynthetic route search for a SMILES"),
    ],
    "admet-mcp": [
        ("predict_bbb_permeability", "predict blood-brain-barrier permeability from SMILES"),
        ("predict_admet", "predict ADMET / toxicity / solubility / CYP endpoints from SMILES"),
    ],
    "bioactivity-mcp": [
        ("fetch_protein_activities", "fetch known actives for a protein from ChEMBL/BindingDB"),
    ],
}

_LOCAL_CSV = pathlib.Path("/tmp/stub_mols.csv")
_STUB_CSV = (
    "Smiles,QED,docking_score\n"
    "O=C1Cc2ccccc2N1C(=O)c1ccc(F)cc1,0.81,-9.2\n"
    "COc1ccc(CNc2ncnc3[nH]ccc23)cc1,0.74,-8.7\n"
    "CC(=O)Nc1ccc(S(=O)(=O)N2CCOCC2)cc1,0.69,-8.1\n"
)


def _write_local_csv() -> None:
    try:
        _LOCAL_CSV.write_text(_STUB_CSV)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Stubbed grounding tools (same NAMES the orchestrator prompt calls). The model
# calls list_available_tools(query) -> frozen inventory -> plans on it.
# ─────────────────────────────────────────────────────────────────────────────
async def list_available_tools(query: str) -> dict:
    """[STUB] Search the MCP tool registry (frozen inventory) for ready-to-use tools."""
    tools = []
    for server, items in INVENTORY.items():
        for name, desc in items:
            tools.append({"name": name, "server_id": server, "description": desc, "score": 0.9})
    return {"status": "success", "count": len(tools), "tools": tools}


async def list_server_tools(server_id: str) -> dict:
    """[STUB] List ALL tools of one MCP server with full descriptions (frozen inventory)."""
    items = INVENTORY.get(server_id)
    if items is None:
        return {"status": "unavailable", "server_id": server_id, "count": 0, "tools": [],
                "message": f"Unknown server '{server_id}'."}
    return {
        "status": "success", "server_id": server_id, "count": len(items),
        "tools": [{"name": n, "description": d, "input_schema": {}, "tags": []} for n, d in items],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stub sub-agents — real AgentTool subclass (name/description byte-identical), but
# returns a canned realistic result instantly; never invokes the real sub-agent.
# ─────────────────────────────────────────────────────────────────────────────
# A realistic presigned-style URL in the TEXT the orchestrator/critic read (so the critic
# does NOT churn on "this is a fake local path"); the finalizer reads the fetchable file://
# url from state['fedot_artifacts'] instead. Keeps the stub trajectory prod-like:
# ground -> generate -> download+read -> synthesize.
_PRESIGNED = ("https://storage.itmo.ru/coscientist/mols_20260614.csv?"
              "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600&X-Amz-Signature=stubsig")
_CANNED = {
    "TaskExecutorAgent": (
        "Ran the experiment via FEDOT.MAS. Generated 12 candidate molecules using "
        f"generate_case_mols. Results uploaded to S3 (presigned URL): {_PRESIGNED} | "
        "columns: Smiles, QED, docking_score | generated_count: 12. The full molecule set "
        "is downloadable at that link."
    ),
    "ResearchAgent": (
        "**Summary** — GSK-3beta (glycogen synthase kinase-3 beta) is a well-established drug "
        "target for Alzheimer's disease (reduces tau hyperphosphorylation). **Key Points** — in "
        "the generative tooling the 'alzheimer' case corresponds to the GSK-3beta target."
    ),
    "CoderAgent": (
        "Sandbox task completed. Downloaded the CSV from the presigned URL and parsed 12 rows. "
        "Sample molecules: O=C1Cc2ccccc2N1C(=O)c1ccc(F)cc1 (QED 0.81, docking -9.2); "
        "COc1ccc(CNc2ncnc3[nH]ccc23)cc1 (QED 0.74, docking -8.7); "
        "CC(=O)Nc1ccc(S(=O)(=O)N2CCOCC2)cc1 (QED 0.69, docking -8.1). All 12 rows extracted "
        "successfully. (No molecules were fabricated — generation is routed to the generation MCP.)"
    ),
    "HypothesesAgent": (
        "Proposed approach: generate candidate molecules for the target case, then score by "
        "docking + ADMET, then filter to the requested constraints."
    ),
    "MedicalAgent": "Clinical summary (stub): literature-based decision support for the query.",
}
_SEED_ARTIFACTS = {"TaskExecutorAgent"}


class StubAgentTool(AgentTool):
    """AgentTool that returns a canned result instantly instead of running the agent.

    Keeps name/description byte-identical to the real tool (AgentTool.__init__ sets
    name=agent.name) so the orchestrator's function_call.name routing, the critic's
    trajectory parsing, and _apply_revisions' request-preservation all behave as in prod.
    """

    async def run_async(self, *, args, tool_context):
        # Mirror ResilientAgentTool's request-injection contract (agents.py:48-51).
        if isinstance(args, dict) and "request" not in args:
            _alt = next((v for v in args.values() if isinstance(v, str) and v.strip()), None)
            args = {**args, "request": _alt or "Proceed with the delegated task."}
        name = self.agent.name
        if name in _SEED_ARTIFACTS:
            tool_context.state["fedot_artifacts"] = [{
                "url": "file:///tmp/stub_mols.csv", "s3_key": "stub/mols.csv",
                "generated_count": 12, "columns": ["Smiles", "QED", "docking_score"],
                "case": "stub",
            }]
        return _CANNED.get(name, f"[stub:{name}] done.")


# ─────────────────────────────────────────────────────────────────────────────
# Design 1 — delegation-gate critic. Fires the LLM critic on the FIRST pending call
# whose name is a roster sub-agent (tag-free, deterministic), once per newly-targeted
# agent; skips internal grounding tools and already-critiqued agents (no churn);
# demotes a domain-doubt REJECT to a logged feedback (bare-text REJECT TERMINATES the
# invocation per ADK event.py — verified by the workflow). Reuses critic_agent helpers.
# ─────────────────────────────────────────────────────────────────────────────
async def delegation_gate_critique(callback_context, llm_response):
    pending = _extract_pending_calls(llm_response)
    if not pending:
        return None
    deleg = [c for c in pending if c["tool"] in ROSTER]
    if not deleg:
        return None  # grounding / internal tool call -> not a plan-in-action turn
    state = callback_context.state
    critiqued = set(state.get("_plan_critiqued_agents", []))
    new_targets = [c for c in deleg if c["tool"] not in critiqued]
    if not new_targets:
        return None  # the plan that delegates to these agents was already critiqued -> skip (no churn)
    state["_plan_critiqued_agents"] = sorted(critiqued | {c["tool"] for c in new_targets})

    contents = _session_contents(callback_context)
    user_task = _first_text(getattr(callback_context, "user_content", None))
    trajectory = _extract_completed_trajectory(contents)
    user_prompt = (
        f"ORIGINAL TASK:\n{user_task}\n\n"
        f"COMPLETED TRAJECTORY:\n{_format_trajectory(trajectory)}\n\n"
        f"PROPOSED NEXT ACTION(S) (the plan in action — not yet executed):\n{_format_pending_calls(pending)}\n\n"
        "Decide whether to approve, revise, or reject these proposed actions. Respond as strict JSON."
    )
    payload = await _invoke_critic_llm(pre_action_critic_instruction, user_prompt)
    verdict = (payload.get("verdict") or "approve").lower().strip()
    feedback = (payload.get("feedback") or "").strip()
    revised = payload.get("revised_calls") or []

    history = state.get("critic_pre_history", [])
    history.append({"verdict": verdict, "feedback": feedback,
                    "proposed": [{"tool": c["tool"], "args": c["args"]} for c in pending],
                    "mode": "delegation"})
    state["critic_pre_history"] = history

    if verdict == PreVerdict.REVISE.value and revised:
        _apply_revisions(pending, revised)
    if verdict == PreVerdict.REJECT.value and llm_response.content and llm_response.content.parts:
        # Demote: do NOT return a terminating bare-text LlmResponse — inject the feedback
        # as a thought and let the orchestrator proceed (it can self-correct next turn).
        llm_response.content.parts.insert(
            0, types.Part(text=f"[CRITIC NOTE — reconsider]: {feedback}", thought=True))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Uniform turn counter wrapper — counts every model turn that proposed tool calls,
# across ALL conditions, so the early-return ratio (firings / turns) is comparable.
# ─────────────────────────────────────────────────────────────────────────────
def make_counting(inner):
    async def cb(callback_context, llm_response):
        calls = _extract_pending_calls(llm_response)
        if calls:
            st = callback_context.state
            st["_turns_with_calls"] = st.get("_turns_with_calls", 0) + 1
        if inner is None:
            return None
        return await inner(callback_context, llm_response)
    return cb


def set_condition(cond: str) -> None:
    """Wire the orchestrator's after_model_callback + settings for a critic mode."""
    s = get_settings()
    s.orchestrator.plan_critic_only = (cond == "tags")
    inner = {
        "none": None,
        "per-action": pre_action_critique,
        "tags": pre_action_critique,
        "delegation": delegation_gate_critique,
    }[cond]
    A.orchestrator_agent.after_model_callback = make_counting(inner)


def install_stubs() -> None:
    """Replace the orchestrator's tools with stubs (grounding + sub-agents)."""
    _write_local_csv()
    stub_subagents = [StubAgentTool(agent=A._resolve_agent(spec.name))
                      for spec in catalog.enabled_agents()]
    A.orchestrator_agent.tools = [
        FunctionTool(list_available_tools),
        FunctionTool(list_server_tools),
        *stub_subagents,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Probe set (subset of the 35-query bank, one per decisive property)
# ─────────────────────────────────────────────────────────────────────────────
PROBES = {
    "ctrl": {
        "prompt": "Generate GSK-3beta inhibitors with high activity",
        "kind": "control",
        "tests": "clean 1-step plan: critic should fire once, approve, no churn",
        "expected_verdict": "approve",
    },
    "fp_alz": {
        "prompt": "Suggest molecules for Alzheimer's disease",
        "kind": "false-positive-trap",
        "tests": "alzheimer=>GSK-3beta case mapping the per-action critic wrongly revised",
        "expected_verdict": "approve",  # any revise/reject of the generation delegation = FALSE POSITIVE
    },
    "gap": {
        "prompt": ("Generate KRAS G12C inhibitors, then run a full clinical-trial simulation "
                   "with Phase II success probability"),
        "kind": "capability-gap",
        "tests": "plan references a capability NOT in the inventory; good critic flags the gap",
        "expected_verdict": "revise_or_reject",  # catching the gap is a TRUE POSITIVE
    },
    "wrong": {
        "prompt": "Write a Python script that generates 10 plausible GSK-3beta SMILES strings",
        "kind": "wrong-agent-fabrication",
        "tests": "bad plan routes to CoderAgent to fabricate; critic should steer away",
        "expected_verdict": "revise_or_reject",
    },
}

CONDITIONS = ["none", "per-action", "tags", "delegation"]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
async def run_one(probe_key: str, cond: str, rep: int, cap: int, stamp: str) -> dict:
    from CoScientist.main import CoScientistManager
    probe = PROBES[probe_key]
    sid = f"pc_{cond}_{probe_key}_{rep}_{stamp}"
    set_condition(cond)
    mgr = CoScientistManager(session_id=sid)
    t0 = time.time()
    err, resp = None, ""
    try:
        resp = await asyncio.wait_for(mgr.run(probe["prompt"], verbose=False), timeout=cap)
    except asyncio.TimeoutError:
        err = f"timeout>{cap}s"
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:200]}"

    # Offline metrics from session state (the critic's own audit trail).
    hist, turns = [], 0
    try:
        sess = await mgr.session_service.get_session(
            app_name=mgr.app_name, user_id=mgr.user_id, session_id=sid)
        st = dict(getattr(sess, "state", {}) or {})
        hist = st.get("critic_pre_history", []) or []
        turns = st.get("_turns_with_calls", 0)
    except Exception:
        pass
    try:
        await mgr.close()
    except Exception:
        pass

    # Did the critic critique a DELEGATION (the plan-in-action), and with what verdict?
    deleg_entries = [h for h in hist if any(p.get("tool") in ROSTER for p in h.get("proposed", []))]
    verdicts = [h.get("verdict") for h in deleg_entries]
    fired_on_plan = len(deleg_entries)
    revise_reject = sum(1 for v in verdicts if v in ("revise", "reject"))
    # False-positive only meaningful on probes whose correct verdict is approve.
    is_fp = (probe["expected_verdict"] == "approve" and revise_reject > 0)
    # True-positive only meaningful on probes that SHOULD be revised/rejected.
    is_tp = (probe["expected_verdict"] == "revise_or_reject" and revise_reject > 0)

    rec = {
        "probe": probe_key, "kind": probe["kind"], "condition": cond, "rep": rep,
        "session_id": sid, "duration_s": round(time.time() - t0, 1),
        "resp_len": len(resp or ""), "error": err,
        "n_critic_firings": len(hist),
        "n_turns_with_calls": turns,
        "fired_on_plan_delegation": fired_on_plan,
        "deleg_verdicts": verdicts,
        "false_positive": is_fp,
        "true_positive": is_tp,
    }
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--probes", default=",".join(PROBES.keys()))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="1 probe (ctrl) x 1 condition (per-action) x 1")
    ap.add_argument("--no-opik", action="store_true", help="skip trace_locator recording")
    args = ap.parse_args()

    install_stubs()

    if args.smoke:
        conditions, probes, repeats = ["per-action"], ["ctrl"], 1
    else:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        probes = [p.strip() for p in args.probes.split(",") if p.strip()]
        repeats = args.repeats

    stamp = datetime.datetime.now().strftime("%H%M%S")
    total = len(conditions) * len(probes) * repeats
    print(f"=== plan-critic probe · model={os.environ['LLM__MAIN_MODEL']} · "
          f"conditions={conditions} · probes={probes} · repeats={repeats} · cap={args.cap}s · "
          f"{total} runs (FEDOT-free, stubbed) ===", flush=True)

    runs = []
    i = 0
    for cond in conditions:
        for pk in probes:
            for rep in range(repeats):
                i += 1
                print(f"[{i}/{total}] cond={cond} probe={pk}", flush=True)
                r = await run_one(pk, cond, rep, args.cap, stamp)
                print(f"    -> {r['duration_s']}s firings={r['n_critic_firings']} "
                      f"plan_firings={r['fired_on_plan_delegation']} turns={r['n_turns_with_calls']} "
                      f"verdicts={r['deleg_verdicts']} FP={r['false_positive']} TP={r['true_positive']} "
                      f"err={r['error']}", flush=True)
                if record_run and not args.no_opik:
                    try:
                        entry = await asyncio.to_thread(
                            record_run, r["session_id"], query=PROBES[pk]["prompt"],
                            condition=cond, model=os.environ["LLM__MAIN_MODEL"])
                        r["trace_id"] = entry["trace_id"] if entry else None
                    except Exception:
                        r["trace_id"] = None
                runs.append(r)

    outdir = REPO / "scripts/experiments/results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"plan_critic_{datetime.date.today().isoformat()}_{stamp}.json"
    out.write_text(json.dumps({"runs": runs, "inventory_servers": list(INVENTORY)},
                              ensure_ascii=False, indent=1))

    # Aggregate summary per condition.
    print("\n=== SUMMARY (per condition) ===", flush=True)
    print(f"{'condition':12} {'runs':>4} {'plan_fire_rate':>14} {'avg_firings':>11} "
          f"{'FP':>3} {'TP':>3} {'avg_s':>6}", flush=True)
    for cond in conditions:
        cr = [r for r in runs if r["condition"] == cond and not r["error"]]
        if not cr:
            print(f"{cond:12} {0:>4}  (all errored/empty)", flush=True)
            continue
        fired = sum(1 for r in cr if r["fired_on_plan_delegation"] > 0)
        fp = sum(1 for r in cr if r["false_positive"])
        tp = sum(1 for r in cr if r["true_positive"])
        avg_fire = round(sum(r["n_critic_firings"] for r in cr) / len(cr), 2)
        avg_s = round(sum(r["duration_s"] for r in cr) / len(cr), 1)
        print(f"{cond:12} {len(cr):>4} {f'{fired}/{len(cr)}':>14} {avg_fire:>11} "
              f"{fp:>3} {tp:>3} {avg_s:>6}", flush=True)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
