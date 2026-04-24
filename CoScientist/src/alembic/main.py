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

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from alembic.agents import explorer_agent, environment_agent, coder_agent, validator_agent
from alembic.tools import WORKDIR

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


def _make_report_guard(required_report: str):
    """Return an after_agent_callback that re-prompts the agent if it forgot write_report.

    Scans session events for the current invocation. If no FunctionResponse from
    write_report with the expected report name is found, returns a Content that
    tells the agent to call write_report before finishing.
    """
    def _guard(callback_context) -> types.Content | None:
        inv_id = callback_context.invocation_id
        for event in callback_context.session.events:
            if event.invocation_id != inv_id:
                continue
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                fr = getattr(part, "function_response", None)
                if fr and fr.name == "write_report":
                    report_path = str((fr.response or {}).get("report_path", ""))
                    if required_report in report_path:
                        return None  # report was written — all good
        # Report missing: instruct the agent to write it now.
        print(f"  [guard] write_report('{required_report}') not found — prompting agent.")
        return types.Content(
            role="user",
            parts=[types.Part(text=(
                f"IMPORTANT: You have not called write_report with "
                f"report_name='{required_report}' yet. "
                f"You MUST call write_report now to save your report before finishing."
            ))],
        )

    return _guard


def _make_venv_guard(venv_python_path: str):
    """Return an after_agent_callback that re-prompts when the venv Python doesn't exist."""
    def _guard(callback_context) -> types.Content | None:
        if Path(venv_python_path).exists():
            return None
        print(f"  [guard] venv not found at {venv_python_path} — prompting agent.")
        return types.Content(
            role="user",
            parts=[types.Part(text=(
                f"IMPORTANT: The virtual environment was not successfully created. "
                f"Expected Python binary at: {venv_python_path}. "
                f"You MUST set up the environment (using setup_venv or bash_env) "
                f"before finishing."
            ))],
        )

    return _guard


def _chain_guards(*guards):
    """Combine multiple after_agent_callbacks into one, stopping at the first non-None result."""
    def _combined(callback_context) -> types.Content | None:
        for g in guards:
            result = g(callback_context)
            if result is not None:
                return result
        return None
    return _combined


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


async def run_agent(
    agent,
    session_service,
    session_id: str,
    message: str,
    required_report: str | None = None,
    venv_guard_path: str | None = None,
) -> str:
    """Run a single agent turn, log every event, return final response text.

    Guards installed via after_agent_callback re-prompt the agent if it finishes
    without satisfying requirements (venv existence and/or write_report).
    """
    # ── install guards ────────────────────────────────────────────────────
    _original_cb = agent.after_agent_callback
    active_guards = []
    if venv_guard_path is not None:
        active_guards.append(_make_venv_guard(venv_guard_path))
    if required_report is not None:
        active_guards.append(_make_report_guard(required_report))
    if active_guards:
        agent.after_agent_callback = (
            active_guards[0] if len(active_guards) == 1
            else _chain_guards(*active_guards)
        )
    # ──────────────────────────────────────────────────────────────────────

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

    finally:
        agent.after_agent_callback = _original_cb

    return final


def _banner(stage: int, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STAGE {stage} — {label}")
    print(f"{'='*60}")


def _clean_workdir(name: str) -> None:
    """Remove the entire work directory for this repo before a fresh run."""
    repo_dir = WORKDIR / name
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
        print(f"  [clean] removed {repo_dir}")


async def run_pipeline(repo_url: str):
    name = _repo_name(repo_url)
    session_service = InMemorySessionService()

    _clean_workdir(name)

    base = WORKDIR / name
    venv_python = str((base / "output" / ".venv" / "bin" / "python").resolve())

    for sid in (f"{name}_explorer", f"{name}_environment",
                f"{name}_coder", f"{name}_validator"):
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=sid
        )

    # ── Stage 1: Explorer ──────────────────────────────────────────────────
    _banner(1, f"Explorer  ({repo_url})")
    await run_agent(explorer_agent, session_service, f"{name}_explorer", repo_url,
                    required_report="exploration")
    print(f"\n[Explorer done] report → {base}/reports/exploration.md")

    # ── Stage 2: Environment + Coder (parallel) ───────────────────────────
    _banner(2, f"Environment + Coder  ({repo_url})")
    await asyncio.gather(
        run_agent(environment_agent, session_service, f"{name}_environment", repo_url,
                  required_report="environment", venv_guard_path=venv_python),
        run_agent(coder_agent, session_service, f"{name}_coder", repo_url,
                  required_report="server"),
    )
    print(f"\n[Environment done] report → {base}/reports/environment.md")
    print(f"[Coder done]       server → {base}/output/server.py")
    print(f"                   tests  → {base}/output/tests/test_server.py")
    print(f"                   report → {base}/reports/server.md")

    # ── Stage 3: Validator (calls Debugger internally on failures) ─────────
    _banner(3, f"Validator  ({repo_url})")
    validator_response = await run_agent(
        validator_agent, session_service, f"{name}_validator", repo_url,
        required_report="validation",
    )
    print(f"\n[Validator done] report → {base}/reports/validation.md")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {name}")
    print(f"  Reports : {base}/reports/")
    print(f"  Output  : {base}/output/")
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
