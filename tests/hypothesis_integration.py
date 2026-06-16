"""
Integration test: HypothesisGenerator (MooseChemTool) + HypothesisCriticAgent.
Verbose logging — prints every step.
"""

from __future__ import annotations

import asyncio, importlib.util, os, sys, time, types as _types, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Verbose logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("test")

# ---- Block CoScientist/__init__.py ----
_mock_cs = _types.ModuleType("CoScientist")
_mock_cs.__path__ = [str(PROJECT_ROOT / "CoScientist")]
sys.modules["CoScientist"] = _mock_cs
for _name in ["fedotmas", "rag_tools"]:
    sys.modules[_name] = _types.ModuleType(_name)

log.info("Blocked CoScientist/__init__.py, fedotmas, rag_tools")

# ---- Load hypothesis_critic.py directly ----
_critic_path = PROJECT_ROOT / "CoScientist" / "agents" / "hypothesis_critic.py"
_spec = importlib.util.spec_from_file_location(
    "CoScientist.agents.hypothesis_critic", str(_critic_path), submodule_search_locations=[])
_critic_mod = importlib.util.module_from_spec(_spec)
sys.modules["CoScientist.agents.hypothesis_critic"] = _critic_mod
_spec.loader.exec_module(_critic_mod)
log.info("Loaded hypothesis_critic.py directly (bypassed agents/__init__.py)")

# ---- Env ----
os.environ.setdefault("OPIK_API_KEY", "test")
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:9999")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

_model = os.getenv("LLM__MAIN_MODEL", "")
_api_key = os.getenv("LLM__OPENAI_API_KEY", "")
_api_url = os.getenv("LLM__MAIN_URL", "")

import litellm
litellm.api_key = _api_key
litellm.api_base = _api_url
log.info(f"LLM config: model={_model}, api_url={_api_url}, key={_api_key[:15]}...")

# ---- Imports ----
log.info("Importing hypothesis_subsystem modules...")
from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.moosechem_tool import MooseChemTool
from CoScientist.hypothesis_subsystem.loop_coordinator import HypothesisLoopCoordinator
from CoScientist.hypothesis_subsystem.models import HypothesisQuery, HypothesisList, HypothesisStatus
log.info("All imports OK")

RESEARCH_QUESTION = (
    "What molecular properties of small organic compounds (50-400 Da) correlate "
    "with their ability to inhibit the main protease (Mpro) of SARS-CoV-2?"
)

async def main():
    audit = HypothesisAuditLogger(logging.getLogger("hypothesis"))

    t0 = time.monotonic()
    log.info("=" * 60)
    log.info("STEP 1: Creating MooseChemTool")
    log.info("=" * 60)
    tool = MooseChemTool(model=_model, max_papers_per_query=2, max_hypotheses=1)
    query = HypothesisQuery(research_question=RESEARCH_QUESTION, max_hypotheses=1)
    log.info(f"Tool created: strategy={tool.strategy_type}, model={_model}")
    log.info(f"Query: {RESEARCH_QUESTION[:100]}...")

    log.info("Calling tool.invoke() — PubMed corpus + LLM generation...")
    t_invoke = time.monotonic()
    result = await tool.invoke(query)
    dt = (time.monotonic() - t_invoke) * 1000
    log.info(f"tool.invoke() returned in {dt:.0f}ms | success={result.success} | hypotheses={len(result.hypotheses)}")
    log.info(f"Metadata: {result.metadata}")

    if not result.success:
        log.error(f"FAIL: {result.error_message}")
        return

    hypotheses = HypothesisList(hypotheses=result.hypotheses)
    log.info("--- GENERATED HYPOTHESIS DUMP ---")
    for i, h in enumerate(hypotheses.hypotheses, 1):
        log.info(f"H#{i} FULL:\n{h.model_dump_json(indent=2)}")
        log.info(f"  H#{i}: {h.claim[:150]}...")
        log.info(f"    domain={h.domain}")
        log.info(f"    vars: {len(h.variables.independent)}I/{len(h.variables.dependent)}D/{len(h.variables.covariates)}C")
        log.info(f"    evidence_basis={len(h.evidence_basis)} refs")
        log.info(f"    strategy_type={h.strategy_type}")
        log.info(f"    status={h.status.value}")

    log.info(f"\nGenerated {len(hypotheses.hypotheses)} hypotheses in {(time.monotonic()-t0)*1000:.0f}ms\n")

    log.info("=" * 60)
    log.info("STEP 2: Creating HypothesisLoopCoordinator + Critic")
    log.info("=" * 60)
    t_critic = time.monotonic()
    coordinator = HypothesisLoopCoordinator(model=_model, audit=audit)
    log.info(f"Coordinator created | max_iterations={coordinator.MAX_ITERATIONS}")
    log.info(f"Critic: HypothesisCriticAgent (model={coordinator._critic._model})")

    log.info("Calling coordinator.run_critic_loop()...")
    refined = await coordinator.run_critic_loop(hypotheses, RESEARCH_QUESTION)
    dt_critic = (time.monotonic() - t_critic) * 1000
    log.info(f"Critic loop done in {dt_critic:.0f}ms | hypotheses={len(refined.hypotheses)}")

    log.info("--- REFINED HYPOTHESIS DUMP ---")
    for i, h in enumerate(refined.hypotheses, 1):
        log.info(f"REFINED H#{i} FULL:\n{h.model_dump_json(indent=2)}")
        icon = {HypothesisStatus.ACTIVE: "PASS", HypothesisStatus.DEFERRED: "DEFER",
                HypothesisStatus.PROPOSED: "NEW", HypothesisStatus.REFUTED: "FAIL",
                HypothesisStatus.CONFIRMED: "DONE"}.get(h.status, "???")
        log.info(f"  [{icon}] H#{i} [{h.status.value}] {h.claim[:120]}...")
        for ev in h.provenance.history:
            log.info(f"    provenance: [{ev.action}] {ev.agent} @ {ev.timestamp.isoformat()}")
            if ev.detail:
                log.info(f"      detail: {ev.detail[:200]}")

    approved = sum(1 for h in refined.hypotheses if h.status == HypothesisStatus.ACTIVE)
    deferred = sum(1 for h in refined.hypotheses if h.status == HypothesisStatus.DEFERRED)
    total_ms = (time.monotonic() - t0) * 1000
    log.info(f"\nTOTAL: {total_ms:.0f}ms | {approved} approved, {deferred} deferred "
             f"(out of {len(refined.hypotheses)})")

if __name__ == "__main__":
    asyncio.run(main())
