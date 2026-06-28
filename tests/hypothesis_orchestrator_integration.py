from __future__ import annotations

# ---- 0. Mock opik BEFORE ANY other import ----
import sys as _sys
import types as _types
for _mod in [
    "opik", "opik.rest_api", "opik.rest_api.core", "opik.rest_api.core.pydantic_utilities",
    "opik.integrations", "opik.integrations.adk", "opik.configurator", "opik.configurator.configure",
]:
    if _mod not in _sys.modules:
        _sys.modules[_mod] = _types.ModuleType(_mod)
_sys.modules["opik"].configure = lambda *a, **kw: None
_sys.modules["opik"].__version__ = "0.0.0"
_sys.modules["opik"].track = lambda name=None: (lambda fn: fn)  # no-op decorator
# Provide OpikTracer so opik_tracer.py doesn't crash
_adk = _sys.modules.setdefault("opik.integrations.adk", _types.ModuleType("opik.integrations.adk"))
_adk.OpikTracer = type("OpikTracer", (), {"__init__": lambda self, *a, **kw: None})
_adk.track_adk_agent_recursive = lambda agent, tracer: agent

"""
Full-cycle integration test - OrchestratorAgent -> HypothesisSubsystem.
Only opik is mocked. All other deps (fedotmas, rag_tools) are real.

Usage:
    .venv\Scripts\python.exe tests/hypothesis_orchestrator_integration.py
"""

import asyncio, logging, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OPIK_API_KEY", "test")
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:9999")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("test.orch")

log.info("Importing CoScientist (real deps, opik mocked)...")
from CoScientist.main import CoScientistManager
from CoScientist.config import get_settings

settings = get_settings()
MODEL = settings.llm.main_model
API_KEY = settings.llm.openai_api_key or settings.llm.service_key
API_URL = settings.llm.main_url

import litellm; litellm.api_key = API_KEY; litellm.api_base = API_URL
log.info(f"LLM: model={MODEL}  api={API_URL}")
log.info("Imports OK. Running full orchestrator test...")
log.info("=" * 70)

RESEARCH_QUESTION = (
    "Generate hypotheses: what molecular properties of small organic "
    "compounds (50-400 Da) correlate with Mpro inhibition of SARS-CoV-2?"
)

async def main():
    t0 = time.monotonic()
    mgr = CoScientistManager(app_name="t", user_id="u", session_id="s")
    result = await mgr.run(RESEARCH_QUESTION, verbose=True)
    dt = (time.monotonic() - t0) / 60.0
    log.info(f"Total: {dt:.1f} min | Result: {len(result)} chars")
    log.info(f"Result: {result[:500]}...")
    await mgr.close()
    if not result or result == "No response":
        log.error("FAIL: empty/default response")
        _sys.exit(1)
    log.info("PASS: Orchestrator -> HypothesisSubsystem")

if __name__ == "__main__":
    asyncio.run(main())
