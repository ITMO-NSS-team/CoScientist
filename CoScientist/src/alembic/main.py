#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import asyncio
import json
import shutil
import textwrap
import traceback
from datetime import datetime

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from alembic.agents import explorer_agent, coder_agent, validator_agent
from alembic.tools import REPORTS_DIR, OUTPUT_DIR

# ── Patch ADK tool lookup to return an error response instead of crashing ─────
# When the LLM hallucinates a tool name, ADK raises ValueError and kills the
# entire agent run.  We replace _get_tool with a version that returns a stub
# tool whose run_async echoes the error back to the LLM so it can self-correct.
import google.adk.flows.llm_flows.functions as _adk_fns

_original_get_tool = _adk_fns._get_tool


class _UnknownToolStub:
    """Returned in place of a missing tool; feeds an error back to the LLM."""
    def __init__(self, called_name: str, available: list):
        self.name = called_name
        self._msg = (
            f"Tool '{called_name}' does not exist. "
            f"You MUST use one of these exact names: "
            f"{', '.join(sorted(available))}. "
            "Retry the call with the correct tool name."
        )

    async def run_async(self, *, args=None, tool_context=None, **_):
        return {"error": self._msg}


def _safe_get_tool(function_call, tools_dict):
    try:
        return _original_get_tool(function_call, tools_dict)
    except (ValueError, KeyError):
        return _UnknownToolStub(function_call.name, list(tools_dict.keys()))


_adk_fns._get_tool = _safe_get_tool
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME = "alembic_app"
USER_ID  = "user_1"

TRUNC = 2000  # max chars shown for tool args / responses inline


def _repo_name(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")


def _trunc(text: str, n: int = TRUNC) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def _log_event(agent_name: str, event) -> None:
    """Print a human-readable line for every ADK event."""
    if not event.content or not event.content.parts:
        return

    for part in event.content.parts:
        if part.text:
            # Agent thinking / final answer
            prefix = "FINAL" if event.is_final_response() else "text"
            snippet = _trunc(part.text.strip())
            print(f"  [{agent_name}] {prefix}: {snippet}")

        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            args_str = _trunc(str(fc.args))
            print(f"  [{agent_name}] CALL  {fc.name}({args_str})")

        elif hasattr(part, "function_response") and part.function_response:
            fr = part.function_response
            resp_str = _trunc(str(fr.response))
            print(f"  [{agent_name}] RESP  {fr.name} → {resp_str}")


MAX_TOOL_REPEATS = 3   # abort if same tool+args combo is called this many times
MAX_STEPS        = 60  # hard ceiling on total events per agent


async def run_agent(agent, session_service, session_id: str, message: str) -> str:
    """Run a single agent turn, log every event, return final response text."""
    runner  = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final   = "Agent did not produce a final response."

    step          = 0
    last_call     = None   # (tool_name, frozen_args) of previous call
    tool_repeats  = 0

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            step += 1
            _log_event(agent.name, event)

            # ── loop / runaway detection ───────────────────────────────────
            if event.content:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc        = part.function_call
                        call_key  = (fc.name, str(fc.args))
                        tool_repeats = tool_repeats + 1 if call_key == last_call else 1
                        last_call    = call_key
                        if tool_repeats >= MAX_TOOL_REPEATS:
                            print(f"  [{agent.name}] ABORT: {fc.name}({_trunc(str(fc.args))}) "
                                  f"called {tool_repeats}x with identical args — breaking loop.")
                            return final

            if step >= MAX_STEPS:
                print(f"  [{agent.name}] ABORT: reached {MAX_STEPS} steps — breaking.")
                return final

            if event.is_final_response():
                if event.content and event.content.parts:
                    final = event.content.parts[0].text or final
                elif event.actions and event.actions.escalate:
                    final = f"Agent escalated: {event.error_message or 'No message.'}"
                break

    except json.JSONDecodeError as e:
        # Model returned a tool-call with invalid JSON arguments (e.g. raw
        # control characters in a long string argument).  Log concisely and
        # continue — the pipeline should not crash over a single bad response.
        print(f"\n  [{agent.name}] WARN: model returned invalid JSON in tool "
              f"call arguments (char {e.pos}): {e.msg} — skipping event.")
        print()

    except Exception as e:
        # Print full traceback so nothing is hidden, then continue pipeline.
        print(f"\n  [{agent.name}] ERROR in event loop:")
        traceback.print_exc()
        print()

    return final


_SNAPSHOT_DIRS = {
    "reports": REPORTS_DIR,
    "output":  OUTPUT_DIR,
}


def _snapshot_tmp(run_dir: Path) -> None:
    """Copy the alembic work dirs into run_dir, overwriting previous snapshot."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for label, src in _SNAPSHOT_DIRS.items():
        if not src.exists():
            continue
        dest = run_dir / label
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)


def _banner(stage: int, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STAGE {stage} — {label}")
    print(f"{'='*60}")


def _clean_workdir(name: str) -> None:
    """Remove output and report files from any previous run for this repo."""
    out_dir = OUTPUT_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"  [clean] removed {out_dir}")

    for report in REPORTS_DIR.glob(f"{name}_*.md"):
        report.unlink()
        print(f"  [clean] removed {report}")


async def run_pipeline(repo_url: str):
    name = _repo_name(repo_url)
    session_service = InMemorySessionService()

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f".alembic/{run_id}-{name}")
    print(f"\n[Run] snapshot dir → {run_dir}")

    _clean_workdir(name)

    for sid in (f"{name}_explorer", f"{name}_coder", f"{name}_validator"):
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=sid
        )

    # ── Stage 1: Explorer ──────────────────────────────────────────────────
    _banner(1, f"Explorer  ({repo_url})")
    explorer_response = await run_agent(
        explorer_agent, session_service, f"{name}_explorer", repo_url
    )
    print(f"\n[Explorer done] report → {REPORTS_DIR}/{name}_exploration.md")
    _snapshot_tmp(run_dir)
    print(f"  [snapshot] {run_dir}")

    # ── Stage 2: Coder ────────────────────────────────────────────────────
    _banner(2, f"Coder  ({repo_url})")
    coder_response = await run_agent(
        coder_agent, session_service, f"{name}_coder", repo_url
    )
    print(f"\n[Coder done] server → {OUTPUT_DIR}/{name}/server.py")
    print(f"             tests  → {OUTPUT_DIR}/{name}/tests/test_server.py")
    _snapshot_tmp(run_dir)
    print(f"  [snapshot] {run_dir}")

    # ── Stage 3: Validator (calls Debugger internally on failures) ─────────
    _banner(3, f"Validator  ({repo_url})")
    validator_response = await run_agent(
        validator_agent, session_service, f"{name}_validator", repo_url
    )
    print(f"\n[Validator done] report → {REPORTS_DIR}/{name}_validation.md")
    _snapshot_tmp(run_dir)
    print(f"  [snapshot] {run_dir}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {name}")
    print(f"  Reports : {REPORTS_DIR}/{name}_*.md")
    print(f"  Output  : {OUTPUT_DIR}/{name}/")
    print(f"{'='*60}")
    print(f"\n--- Validator summary ---\n")
    print(textwrap.indent(validator_response.strip(), "  "))
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./main.py <repo_url>")
        print("Example: ./main.py https://github.com/Roestlab/massformer")
        sys.exit(1)

    repo_url = sys.argv[1]
    try:
        asyncio.run(run_pipeline(repo_url))
    except Exception as e:
        print(f"\nPipeline error: {e}")
        raise
