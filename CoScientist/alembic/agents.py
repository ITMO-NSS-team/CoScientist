import asyncio
import contextvars
import os

import litellm
from loguru import logger

litellm.suppress_debug_info = True

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from alembic.agent_runtime import _trunc
from alembic.tools import (
    clone_repo, read_file, bash, bash_env, search,
    read_report, write_report,
    write_file, read_output_file, update_file,
    validate_syntax, run_tests, setup_venv, check_venv_compat,
    invoke_mcp_tool,
)
from alembic.instructions import (
    explorer_instruction, coder_instruction,
    debugger_instruction, validator_instruction,
    environment_instruction,
)
MODEL = os.environ.get("MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")

# Set once per pipeline run (main.py, before any stage) so the debugger
# AgentTool wrapper below can always stamp the repo URL onto its calls,
# regardless of what the validator LLM remembers to include (F15).
_current_repo_url: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_repo_url", default=""
)


def set_current_repo_url(repo_url: str) -> None:
    _current_repo_url.set(repo_url)


# Bounds a single debugger call so a stuck LLM/tool turn can't consume the
# whole Validator stage budget (F16). Generous enough for a normal
# multi-step fix (a couple of bash_env installs + an edit + a re-invoke)
# while leaving room for several calls within the 1800s validator timeout.
DEBUGGER_CALL_TIMEOUT = 600  # seconds

# F29: the debugger's own tool calls/reasoning happen inside a nested
# ADK sub-Runner (see AgentTool.run_async) whose events never pass through
# agent_runtime._log_event — only the debugger's final self-reported text
# surfaces to the validator's log, as one opaque `RESP debugger -> {...}`
# line. This made a real, minutes-long debugger call impossible to audit
# after the fact (e.g. what bash commands it ran, what it actually edited).
# _debug_round tags every debugger-internal log line with which top-level
# validator->debugger call it belongs to, since the validator can call the
# debugger many times across one stage and the callbacks below (attached to
# debugger_agent, so they fire regardless of nesting) have no other way to
# tell which call is currently in flight.
_debug_round: contextvars.ContextVar[int] = contextvars.ContextVar(
    "debug_round", default=0
)


def _debugger_before_tool(tool, args, tool_context):
    round_num = _debug_round.get()
    logger.debug(f"[debugger#{round_num}] CALL  {tool.name}({_trunc(str(args))})")
    return None


def _debugger_after_tool(tool, args, tool_context, tool_response):
    round_num = _debug_round.get()
    logger.debug(f"[debugger#{round_num}] RESP  {tool.name} → {_trunc(str(tool_response))}")
    return None


def _debugger_after_model(callback_context, llm_response):
    round_num = _debug_round.get()
    content = getattr(llm_response, "content", None)
    if content and content.parts:
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                logger.debug(f"[debugger#{round_num}] text:  {_trunc(text.strip())}")
    return None


class _DebuggerAgentTool(AgentTool):
    """Hardens the raw AgentTool call into the debugger sub-agent.

    The AgentTool's generic ``{"request": str}`` schema has no field that
    forces the repo URL to be present, and the validator has been observed
    omitting it, leaving the debugger unable to locate the repo at all
    (F15). It has also been observed hitting the full stage timeout stuck
    on one call (F16) and burning an attempt on a transient provider error
    with zero diagnostic value (F17). See docs/IMPROVEMENTS_SPEC.md.
    """

    def __init__(self, *, agent):
        super().__init__(agent=agent)
        self._round = 0  # F29: per-process debug-round counter, see above.

    async def run_async(self, *, args, tool_context):
        self._round += 1
        round_num = self._round
        _debug_round.set(round_num)

        repo_url = _current_repo_url.get()
        request = (args.get("request") or "").strip()
        if repo_url:
            request = f"Repository: {repo_url}\n\n{request}"
        call_args = {**args, "request": request}

        logger.info(f"[debugger#{round_num}] ── round {round_num} start ──")
        try:
            for attempt in range(2):
                try:
                    result = await asyncio.wait_for(
                        super().run_async(args=call_args, tool_context=tool_context),
                        timeout=DEBUGGER_CALL_TIMEOUT,
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[debugger#{round_num}] call timed out after {DEBUGGER_CALL_TIMEOUT}s"
                    )
                    return (
                        f"Debugger call timed out after {DEBUGGER_CALL_TIMEOUT}s "
                        "without a resolution. Treat this as an unresolved "
                        "failure for this stage/tool and move on."
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if attempt == 0:
                        logger.warning(
                            f"[debugger#{round_num}] transient error ({e!r}), retrying once"
                        )
                        continue
                    logger.warning(f"[debugger#{round_num}] error persisted after retry: {e!r}")
                    return (
                        f"Debugger call failed twice ({type(e).__name__}: {e}) "
                        "with no diagnostic content. Treat this as an unresolved "
                        "failure for this stage/tool and move on."
                    )
        finally:
            logger.info(f"[debugger#{round_num}] ── round {round_num} end ──")


explorer_agent = Agent(
    name="explorer",
    model=LiteLlm(model=MODEL),
    description="Clones a scientific GitHub repo and writes a Markdown report of its functionality and MCP usage scenarios.",
    instruction=explorer_instruction,
    tools=[clone_repo, read_file, bash, search, write_report],
)

environment_agent = Agent(
    name="environment",
    model=LiteLlm(model=MODEL),
    description="Reads the explorer report and sets up the Python virtual environment for the repository, retrying until successful.",
    instruction=environment_instruction,
    tools=[read_report, setup_venv, bash_env, check_venv_compat, write_report],
)

coder_agent = Agent(
    name="coder",
    model=LiteLlm(model=MODEL),
    description="Reads an explorer report and implements a FastMCP server with pytest tests for the repository.",
    instruction=coder_instruction,
    tools=[read_report, bash, read_file, write_file, write_report],
)

debugger_agent = Agent(
    name="debugger",
    model=LiteLlm(model=MODEL),
    description="Receives a repo URL and an error message, fixes the bug — either by installing a missing system/pip dep or by editing server.py/helpers — and re-runs the failing tool to confirm.",
    instruction=debugger_instruction,
    tools=[read_output_file, update_file, bash, bash_env, invoke_mcp_tool],
    # F29: log the debugger's own steps (which are otherwise invisible to
    # the pipeline log — see _debug_round above) via callbacks, since those
    # fire on every tool/model call regardless of whether this agent is run
    # top-level or nested inside _DebuggerAgentTool's AgentTool.run_async.
    before_tool_callback=_debugger_before_tool,
    after_tool_callback=_debugger_after_tool,
    after_model_callback=_debugger_after_model,
)

validator_agent = Agent(
    name="validator",
    model=LiteLlm(model=MODEL),
    description="Validates the generated MCP server via syntax checks, pytest, and real tool invocations, calling the debugger agent on failures, then writes a validation report.",
    instruction=validator_instruction,
    tools=[read_report, validate_syntax, run_tests, invoke_mcp_tool, write_report, _DebuggerAgentTool(agent=debugger_agent)],
)
