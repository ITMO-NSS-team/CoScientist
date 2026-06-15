"""submit_plan FunctionTool (R09) — the plan-object seam shared by BOTH bridges.

The orchestrator calls submit_plan(plan) BEFORE delegating. The handler:
  1. parses the raw dict into an ExperimentPlan (do NOT trust the type annotation —
     ADK FunctionTool passes a raw dict and swallows pydantic errors, so we validate
     explicitly here);
  2. runs the deterministic gate (R12, experiments/gate.py) against the inventory;
  3. (optional) raises HITL approve/EDIT, RE-VALIDATING an EDITed plan through the
     same gate before accepting it (SessionAgent reads edits back verbatim with no
     check — we close that hole here);
  4. stores the approved+validated plan in state['experiment_plan'] for the bridge;
  5. returns a machine result {accepted, gate_code, detail, n_steps, order}.

run_submit_plan is the testable pure core; make_submit_plan_tool wraps it for ADK.
"""
from __future__ import annotations

from typing import Any, Optional

from google.adk.tools import FunctionTool

from CoScientist.experiments.plan import ExperimentPlan
from CoScientist.experiments.gate import deterministic_gate


async def run_submit_plan(
    plan: Any,
    inventory: Any,
    *,
    hitl_handler: Any = None,
    plan_file_path: Optional[str] = None,
    state: Any = None,
    verbose_errors: bool = True,
) -> dict:
    """Testable core: validate -> gate -> (HITL approve/EDIT, re-validated) -> store.

    verbose_errors (fidelity fix): when True, a malformed plan returns the EXACT pydantic
    error so the model can fix it on the next call; when False, a generic message (the
    pre-fix behavior, kept for the A/B comparison run).
    """
    # 1. parse (explicit — never trust the annotation)
    try:
        p = ExperimentPlan(**plan) if isinstance(plan, dict) else ExperimentPlan.model_validate(plan)
    except Exception as e:  # PlanError or pydantic ValidationError
        detail = (f"plan failed schema validation — fix and resubmit: {type(e).__name__}: {str(e)[:500]}"
                  if verbose_errors
                  else "plan did not match the required schema; resubmit valid JSON")
        return {"accepted": False, "gate_code": "reject:malformed", "detail": detail,
                "n_steps": 0, "order": []}

    # 2. deterministic gate (model-free)
    g = deterministic_gate(p, inventory)
    if not g.ok:
        return {"accepted": False, "gate_code": g.code, "detail": g.detail,
                "n_steps": len(p.steps), "order": [s.id for s in p.steps]}

    # 3. optional HITL approve / EDIT loop — re-validate any edit through the gate
    edit_revalidation_caught = False
    if hitl_handler is not None:
        from CoScientist.hitl.models import HITLRequest, HITLAction
        for _ in range(3):
            if plan_file_path:
                try:
                    with open(plan_file_path, "w", encoding="utf-8") as f:
                        f.write(p.model_dump_json(indent=2))
                except Exception:
                    pass
            resp = await hitl_handler.handle_request(HITLRequest(
                agent_name="submit_plan", action_type=HITLAction.APPROVE, invoked_via="tool",
                message="Review/EDIT the experiment plan (roadmap) before execution.",
                context={"output": p.model_dump_json(indent=2)}))
            edited = (resp.instructions or resp.free_input) if resp else None
            if resp and resp.approved and not edited:
                break
            if not edited:
                break  # rejected with no edit -> keep last valid plan
            # RE-VALIDATE the human edit (close the SessionAgent read-back hole)
            try:
                p2 = ExperimentPlan.model_validate_json(edited)
            except Exception:
                edit_revalidation_caught = True
                continue  # malformed edit -> re-prompt
            g2 = deterministic_gate(p2, inventory)
            if g2.ok:
                p = p2  # accept the validated edit; loop once more to confirm
                continue
            edit_revalidation_caught = True  # edit reintroduced a gate violation -> re-prompt

    # 4. store for the bridge
    if state is not None:
        try:
            state["experiment_plan"] = p.model_dump()
        except Exception:
            pass

    return {"accepted": True, "gate_code": "pass", "detail": "plan accepted",
            "n_steps": len(p.steps), "order": [s.id for s in p.topological_order()],
            "edit_revalidation_caught": edit_revalidation_caught}


# Two descriptions for the tool. The RICH one (fidelity fix) gives the model the exact
# schema + a worked example so it nails the JSON on the first call; the TERSE one is the
# pre-fix behavior kept for the A/B comparison.
_RICH_DOC = '''Submit your FULL experiment plan as a JSON DAG, BEFORE delegating to any sub-agent. Call this FIRST.

Schema:
{"goal": "<one line>",
 "steps": [{"id": "s1", "subtask": "<imperative sub-task>",
            "kind": "compute|research|hypothesize|code_exec",
            "tool_servers": [{"server": "<exact server from list_available_tools>", "tools": ["<exact tool>"]}],
            "run_params": {"<k>": "<v or {artifact_id}>"},
            "expected_artifacts": [{"id": "<stable id>", "description": "<what it holds>"}],
            "deps": ["<prerequisite step id>"]}]}

Rules: a 'compute' step MUST list >=1 tool_server whose tools you actually saw via list_available_tools;
research/hypothesize/code_exec steps may omit tool_servers; wire data flow with deps + {artifact_id}.

Example:
{"goal":"selective KRAS G12C inhibitors",
 "steps":[{"id":"s1","subtask":"generate KRAS G12C candidates","kind":"compute",
           "tool_servers":[{"server":"generative-models-mcp","tools":["generate_case_mols"]}],
           "expected_artifacts":[{"id":"mols","description":"candidate molecules"}]},
          {"id":"s2","subtask":"dock vs KRAS, HRAS, NRAS","kind":"compute","deps":["s1"],
           "tool_servers":[{"server":"chemical-mcp-server","tools":["calculate_docking"]}],
           "run_params":{"input":"{mols}"},"expected_artifacts":[{"id":"docked","description":"scores"}]}]}

Returns {"accepted": bool, "gate_code": str, "detail": str, "order": [str]}. If accepted is False,
FIX the plan exactly per `detail` and call submit_plan again.'''

_TERSE_DOC = '''Submit the experiment plan (a JSON DAG of steps) for validation BEFORE any delegation.
Returns {"accepted": bool, "gate_code": str, "detail": str, "order": [str]}. If not accepted, fix and resubmit.'''


def make_submit_plan_tool(inventory, hitl_handler=None, plan_file_path=None,
                          fidelity_fix: bool = True) -> FunctionTool:
    """fidelity_fix=True: rich schema+example description AND exact validation errors (the fix).
    fidelity_fix=False: terse description + generic errors (pre-fix baseline for the A/B run)."""
    async def submit_plan(plan: dict, tool_context=None) -> dict:
        state = tool_context.state if tool_context is not None else None
        return await run_submit_plan(plan, inventory, hitl_handler=hitl_handler,
                                     plan_file_path=plan_file_path, state=state,
                                     verbose_errors=fidelity_fix)
    submit_plan.__doc__ = _RICH_DOC if fidelity_fix else _TERSE_DOC
    return FunctionTool(submit_plan)
