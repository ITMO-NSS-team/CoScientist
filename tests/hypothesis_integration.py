"""
Isolated integration test: HypothesisGenerator + Critic loop.
Reads test_cases.json, runs all cases, saves structured output.

Usage:
    .venv\Scripts\python.exe tests/hypothesis_integration.py [--cases test_cases.json]
"""

from __future__ import annotations

import argparse, asyncio, importlib.util, json, logging, os, sys, time, types as _types
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Block CoScientist/__init__.py + fedotmas/rag_tools ----
_mock_cs = _types.ModuleType("CoScientist")
_mock_cs.__path__ = [str(PROJECT_ROOT / "CoScientist")]
sys.modules["CoScientist"] = _mock_cs
for _name in ["fedotmas", "rag_tools"]:
    sys.modules[_name] = _types.ModuleType(_name)

# ---- Load hypothesis_critic.py directly ----
_critic_path = PROJECT_ROOT / "CoScientist" / "agents" / "hypothesis_critic.py"
_spec = importlib.util.spec_from_file_location(
    "CoScientist.agents.hypothesis_critic", str(_critic_path), submodule_search_locations=[])
_critic_mod = importlib.util.module_from_spec(_spec)
sys.modules["CoScientist.agents.hypothesis_critic"] = _critic_mod
_spec.loader.exec_module(_critic_mod)

# ---- Env ----
os.environ.setdefault("OPIK_API_KEY", "test")
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:9999")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

_MODEL = os.getenv("LLM__MAIN_MODEL", "")
_API_KEY = os.getenv("LLM__OPENAI_API_KEY", "")
_API_URL = os.getenv("LLM__MAIN_URL", "")

import litellm
litellm.api_key = _API_KEY
litellm.api_base = _API_URL

# ---- Imports ----
from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.moosechem_tool import MooseChemTool
from CoScientist.hypothesis_subsystem.loop_coordinator import HypothesisLoopCoordinator
from CoScientist.hypothesis_subsystem.models import HypothesisQuery, HypothesisList, HypothesisStatus

# ---- Helpers ----

def _extract_scores(h) -> dict:
    """Extract critic scores from provenance history."""
    scores = {"verifiability": None, "consistency": None, "specificity": None, "novelty": None}
    for ev in h.provenance.history:
        if ev.action == "critiqued" and ev.detail and "Passed: scores=" in ev.detail:
            try:
                raw = ev.detail.split("Passed: scores=")[1]
                scores.update(eval(raw))
            except Exception:
                pass
    return scores


async def run_one_case(case: dict, case_index: int) -> dict:
    """Run a single case through Generator+Critic. Returns summary dict."""
    case_id = case.get("id", f"case_{case_index}")
    question = case["research_question"]
    max_hyps = case.get("max_hypotheses", 1)

    # ---- Per-case output dir + logging ----
    case_dir = OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Dual logging: console + file
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(str(case_dir / "orchestrator.log"), encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    console_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    log = logging.getLogger("test")
    audit = HypothesisAuditLogger(logging.getLogger("hypothesis"))

    summary = {
        "case_id": case_id,
        "question": question,
        "model": _MODEL,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "duration_sec": 0.0,
        "hypotheses_count": 0, "approved": 0, "deferred": 0, "proposed": 0,
        "critic_scores": [],
        "hypotheses": [],
        "error": None,
    }

    try:
        t0 = time.monotonic()

        # STEP 1: MooseChemTool
        log.info("=" * 60)
        log.info(f"Case [{case_id}]: {question[:100]}...")
        log.info("=" * 60)

        tool = MooseChemTool(model=_MODEL, max_papers_per_query=2, max_hypotheses=max_hyps)
        query = HypothesisQuery(research_question=question, max_hypotheses=max_hyps)
        log.info(f"Tool: strategy={tool.strategy_type}, model={_MODEL}")

        log.info("Calling MooseChemTool.invoke()...")
        t_invoke = time.monotonic()
        result = await tool.invoke(query)
        log.info(f"invoke() done in {(time.monotonic()-t_invoke)*1000:.0f}ms | "
                 f"success={result.success} | hyps={len(result.hypotheses)}")

        if not result.success:
            summary["status"] = "FAIL"
            summary["error"] = result.error_message or "MooseChemTool failed"
            root.removeHandler(console_handler); root.removeHandler(file_handler)
            return summary

        hypotheses = HypothesisList(hypotheses=result.hypotheses)

        log.info("--- GENERATED HYPOTHESES ---")
        for i, h in enumerate(hypotheses.hypotheses, 1):
            log.info(f"H#{i}: {h.claim[:150]}...")
            log.info(f"  domain={h.domain}, vars={len(h.variables.independent)}I/"
                     f"{len(h.variables.dependent)}D/{len(h.variables.covariates)}C, "
                     f"refs={len(h.evidence_basis)}")

        # STEP 2: Critic loop
        log.info("=" * 60)
        log.info("STEP 2: Critic loop")
        log.info("=" * 60)

        coordinator = HypothesisLoopCoordinator(model=_MODEL, audit=audit)
        log.info(f"Coordinator: max_iter={coordinator.MAX_ITERATIONS}, "
                 f"critic={coordinator._critic._model}")

        t_critic = time.monotonic()
        refined = await coordinator.run_critic_loop(hypotheses, question)
        log.info(f"Critic loop done in {(time.monotonic()-t_critic)*1000:.0f}ms")

        log.info("--- REFINED HYPOTHESES ---")
        for i, h in enumerate(refined.hypotheses, 1):
            icon = {HypothesisStatus.ACTIVE: "PASS", HypothesisStatus.DEFERRED: "DEFER",
                    HypothesisStatus.PROPOSED: "NEW"}.get(h.status, "???")
            scores = _extract_scores(h)
            log.info(f"[{icon}] H#{i} [{h.status.value}] {h.claim[:120]}...")
            log.info(f"  Scores: {scores}")
            for ev in h.provenance.history:
                log.info(f"  provenance: [{ev.action}] {ev.agent}: "
                         f"{ev.detail[:120] if ev.detail else '-'}")

            hy_summary = {
                "claim": h.claim,
                "status": h.status.value,
                "strategy": h.strategy_type,
                "scores": scores,
                "domain": h.domain,
                "refutation": h.refutation_conditions[:300],
                "provenance_events": len(h.provenance.history),
            }
            summary["hypotheses"].append(hy_summary)

            status_key = {"active": "approved", "proposed": "proposed",
                          "deferred": "deferred"}.get(h.status.value, "proposed")
            summary[status_key] += 1
            if scores.get("verifiability") is not None:
                summary["critic_scores"].append(scores)

        summary["hypotheses_count"] = len(refined.hypotheses)
        summary["duration_sec"] = round(time.monotonic() - t0, 1)

    except Exception as exc:
        summary["status"] = "ERROR"
        summary["error"] = str(exc)
        log.error(f"Case [{case_id}] FAILED: {exc}")

    # ---- Save hypotheses.json ----
    with open(case_dir / "hypotheses.json", "w", encoding="utf-8") as f:
        json.dump(summary["hypotheses"], f, ensure_ascii=False, indent=2, default=str)

    # ---- Save summary.json ----
    with open(case_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    root.removeHandler(console_handler)
    root.removeHandler(file_handler)
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="tests/test_cases.json")
    args = parser.parse_args()

    cases_path = PROJECT_ROOT / args.cases
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Model: {_MODEL}")
    print(f"Cases: {len(cases)}  Output: {OUTPUT_DIR}")
    print("=" * 70)

    all_summaries = []
    total_start = time.monotonic()

    for i, case in enumerate(cases, 1):
        case_id = case.get("id", f"case_{i}")
        print(f"\n[{i}/{len(cases)}] {case_id}: {case['research_question'][:80]}...")

        case_start = time.monotonic()
        summary = await run_one_case(case, i)
        dt = time.monotonic() - case_start

        icon = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR"}.get(summary["status"], "???")
        print(f"  [{icon}] {summary['status']} | {summary['hypotheses_count']} hyps | "
              f"{summary['approved']} appr, {summary['deferred']} def | {dt:.0f}s")
        if summary["error"]:
            print(f"  Error: {summary['error']}")

        all_summaries.append({
            "case_id": summary["case_id"], "status": summary["status"],
            "duration_sec": summary["duration_sec"],
            "hypotheses_count": summary["hypotheses_count"],
            "approved": summary["approved"], "deferred": summary["deferred"],
            "proposed": summary["proposed"],
            "critic_scores": summary.get("critic_scores", []),
            "error": summary["error"],
        })

    run_summary = {
        "model": _MODEL,
        "total_duration_sec": round(time.monotonic() - total_start, 1),
        "cases_total": len(cases),
        "passed": sum(1 for s in all_summaries if s["status"] == "PASS"),
        "failed": sum(1 for s in all_summaries if s["status"] != "PASS"),
        "cases": all_summaries,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDONE. {run_summary['passed']}/{run_summary['cases_total']} passed "
          f"({run_summary['total_duration_sec']:.0f}s)")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
