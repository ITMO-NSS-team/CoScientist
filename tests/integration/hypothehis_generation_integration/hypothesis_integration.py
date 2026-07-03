"""
Isolated integration test: HypothesisGenerator + Critic loop.

Uses the full hypothesis_subsystem pipeline (MooseChemTool →
LoopCoordinator → HypothesisCriticAgent) as a black box via
build_hypothesis_subsystem().

Reads test_cases.json, runs all cases, saves structured output.

Usage:
    .venv\Scripts\python.exe tests/integration/hypothehis_generation_integration/hypothesis_integration.py [--cases test_cases.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Opik (best-effort) ----
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

# ---- Subsystem imports ----
from CoScientist.hypothesis_subsystem import build_hypothesis_subsystem
from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.models import HypothesisStatus


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
    """Run a single case through the full subsystem (Generator+Critic)."""
    case_id = case.get("id", f"case_{case_index}")
    question = case["research_question"]

    # ---- Per-case output dir + logging ----
    case_dir = OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

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

        log.info("=" * 60)
        log.info(f"Case [{case_id}]: {question[:100]}...")
        log.info("=" * 60)

        # Build the subsystem (single black-box AgentTool)
        subsystem = build_hypothesis_subsystem(model=_MODEL)
        log.info(f"Subsystem built: agent={subsystem.agent.name}, model={_MODEL}")

        # Run the subsystem agent directly via ADK runner
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="test", user_id="integration_test", session_id=case_id,
        )
        runner = Runner(
            agent=subsystem.agent,
            app_name="test",
            session_service=session_service,
        )

        log.info("Running hypothesis subsystem...")
        t_run = time.monotonic()

        final_response = ""
        async for event in runner.run_async(
            user_id="integration_test",
            session_id=case_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=question)],
            ),
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    parts = event.content.parts
                    final_response = "\n".join(
                        p.text for p in parts
                        if getattr(p, "text", None) and not getattr(p, "thought", False)
                    ) or ""
            # Collect generated hypotheses from state
            session = await session_service.get_session(
                app_name="test", user_id="integration_test", session_id=case_id,
            )
            state = getattr(session, "state", {}) or {}
            generated = state.get("generated_hypotheses")

        log.info(f"Subsystem run done in {(time.monotonic() - t_run) * 1000:.0f}ms")

        # Parse generated hypotheses from state
        hypotheses = []
        if generated and isinstance(generated, dict):
            raw_hyps = generated.get("hypotheses", [])
            from CoScientist.hypothesis_subsystem.models import Hypothesis
            for h in raw_hyps:
                try:
                    hypotheses.append(Hypothesis(**h) if isinstance(h, dict) else h)
                except Exception:
                    pass

        if not hypotheses:
            # Fallback: try parsing final_response as HypothesisList JSON
            try:
                import json as _json
                parsed = _json.loads(final_response) if isinstance(final_response, str) else {}
                raw_hyps = parsed.get("hypotheses", [])
                from CoScientist.hypothesis_subsystem.models import Hypothesis
                for h in raw_hyps:
                    try:
                        hypotheses.append(Hypothesis(**h) if isinstance(h, dict) else h)
                    except Exception:
                        pass
            except Exception:
                pass

        log.info("--- OUTPUT HYPOTHESES ---")
        for i, h in enumerate(hypotheses, 1):
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

        summary["hypotheses_count"] = len(hypotheses)
        summary["duration_sec"] = round(time.monotonic() - t0, 1)

        if not hypotheses:
            summary["status"] = "FAIL"
            summary["error"] = "No hypotheses produced"

    except Exception as exc:
        summary["status"] = "ERROR"
        summary["error"] = str(exc)
        log.error(f"Case [{case_id}] FAILED: {exc}")

    # ---- Save outputs ----
    with open(case_dir / "hypotheses.json", "w", encoding="utf-8") as f:
        json.dump(summary["hypotheses"], f, ensure_ascii=False, indent=2, default=str)

    with open(case_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    root.removeHandler(console_handler)
    root.removeHandler(file_handler)
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "tests" / "test_cases.json"))
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        # Fallback: try relative to integration test directory
        cases_path = Path(__file__).resolve().parent.parent / "test_cases.json"
    if not cases_path.exists():
        print(f"ERROR: test_cases.json not found at {cases_path}")
        print("Creating default test case...")
        cases = [{"id": "default", "research_question": "What molecular features predict EGFR kinase inhibition?", "max_hypotheses": 2}]
    else:
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
