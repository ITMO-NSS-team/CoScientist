"""
CoScientist - Main entry point

Runs the multi-agent scientific discovery pipeline:
- Hypothesis generation
- Research
- Experimentation (FEDOT)
- Orchestration
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import Optional
import logging

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from CoScientist.config import get_settings, ReportConfig
from CoScientist.agents import (
    orchestrator_agent,
    root_agent,
    pipeline_pre_agents,
    pipeline_post_agents,
)
from CoScientist.reporting import finalize_report, RunResult
from CoScientist.agents.callbacks import cleanup_uploaded_papers
from CoScientist.hitl.tool import hitl_toolset
from CoScientist.hitl import (
    AbstractHITLHandler,
    HITLRequest,
    HITLResponse,
)

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directive handed to post-pipeline stages (e.g. the Result Aggregator). They run
# over the SAME session as the orchestrator, so all results are already in state
# and history — they just need the cue to compile the deliverable.
_POST_STAGE_DIRECTIVE = (
    "The run is complete. Using the results already in session state and the sandbox "
    "workspace, produce the final report now. Call format_results first, then write the "
    "full Markdown report as your response."
)


def _s3_csv_preview(url: str, max_rows: int = 10, max_bytes: int = 200_000) -> str:
    """Best-effort: download a presigned-S3 CSV and return a small text preview
    (header + first rows of Smiles + key property columns). Returns '' on any failure.

    Lets the final answer be formed from the ACTUAL S3 file contents rather than a bare
    link or unverified prose (F010.A6).
    """
    import urllib.request
    import csv
    import io
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read(max_bytes).decode("utf-8", "replace")
        rows = list(csv.reader(io.StringIO(raw)))
        if not rows:
            return ""
        hdr = rows[0]
        prefer = ("Smiles", "QED", "LogP", "Synthetic Accessibility", "Validity")
        keep = [i for i, h in enumerate(hdr) if h in prefer] or list(range(min(5, len(hdr))))
        out = [" | ".join(hdr[i] for i in keep)]
        for row in rows[1:1 + max_rows]:
            out.append(" | ".join(row[i] if i < len(row) else "" for i in keep))
        extra = len(rows) - 1 - max_rows
        if extra > 0:
            out.append(f"… (+{extra} more rows)")
        return "\n".join(out)
    except Exception:
        return ""


class CoScientistManager:
    """
    Main manager for CoScientist (ADK-based execution).
    """

    def __init__(
        self,
        app_name: str = "coscientist_app",
        user_id: str = "user_1",
        session_id: str = "session_001",
        hitl_handler: Optional[AbstractHITLHandler] = None,
    ):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id

        self.session_service: Optional[InMemorySessionService] = None
        self.runner: Optional[Runner] = None
        self._runners: dict = {}  # agent name -> Runner (shared session_service)
        self._run_error: Optional[Exception] = None
        self._initialized = False

        # HITL setup
        self._hitl_handler = hitl_handler


    async def initialize(self):
        """Initialize session + runner."""
        if self._initialized:
            return
    
        # Session service
        self.session_service = InMemorySessionService()

        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )

        # Runner over the root orchestrator. Pipeline-stage agents get their own
        # runners (see _runner_for), all sharing this session_service so state
        # flows between stages.
        self.runner = Runner(
            agent=root_agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )
        self._runners[root_agent.name] = self.runner

        if self._hitl_handler:
            hitl_toolset._handler = self._hitl_handler

        self._initialized = True

    def _runner_for(self, agent) -> Runner:
        """A Runner for a pipeline-stage agent, sharing the root's session."""
        runner = self._runners.get(agent.name)
        if runner is None:
            runner = Runner(
                agent=agent,
                app_name=self.app_name,
                session_service=self.session_service,
            )
            self._runners[agent.name] = runner
        return runner

    async def _set_state(self, key: str, value) -> None:
        """Best-effort write into the live session state (in-memory)."""
        try:
            session = await self.session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id,
            )
            if session is not None:
                session.state[key] = value
        except Exception as exc:
            logger.warning("could not set session state %r: %s", key, exc)

    async def run_pipeline(self, query: str, report_config: ReportConfig, verbose: bool = False):
        """Drive pre → root → post stages over one shared session.

        Yields ``(stage_name, event)`` for every event. A stage that raises does
        not abort the pipeline: the error is recorded on ``self._run_error`` and
        remaining stages (notably the post/report stage) still run, so a partial
        failure still yields a report.
        """
        await self.initialize()
        await self._set_state("report_config", report_config.to_state())
        self._run_error = None

        stages = (
            [(a, query) for a in pipeline_pre_agents]
            + [(root_agent, query)]
            + [(a, _POST_STAGE_DIRECTIVE) for a in pipeline_post_agents]
        )
        for agent, message in stages:
            content = types.Content(role="user", parts=[types.Part(text=message)])
            try:
                async for event in self._runner_for(agent).run_async(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    new_message=content,
                ):
                    if verbose:
                        print(
                            f"[Event] {agent.name} | {event.author} | "
                            f"{type(event).__name__} | Final={event.is_final_response()}"
                        )
                    yield agent.name, event
            except Exception as exc:
                self._run_error = exc
                logger.error(
                    "pipeline stage %s raised (%s: %s); continuing to next stage.",
                    agent.name, type(exc).__name__, str(exc)[:200],
                )

    @staticmethod
    def _final_text(event) -> Optional[str]:
        """Answer text of a final-response event, skipping thinking parts."""
        if not (event.is_final_response() and event.content and event.content.parts):
            if event.actions and event.actions.escalate:
                return f"Escalation: {getattr(event, 'error_message', None) or 'Unknown error'}"
            return None
        parts = event.content.parts
        # Thinking models emit a separate `thought` part before the answer; prefer
        # the non-thought text, falling back to any text so we never drop it.
        answer = "\n".join(
            p.text for p in parts
            if getattr(p, "text", None) and not getattr(p, "thought", False)
        )
        return answer or "\n".join(p.text for p in parts if getattr(p, "text", None)) or None

    async def run(
        self,
        query: str,
        verbose: bool = True,
        report_config: Optional[ReportConfig] = None,
    ) -> RunResult:
        """Run the full pipeline (pre → orchestrator → post) and package the report.

        Returns a :class:`RunResult` (``markdown``, ``report_dir``, ``manifest``).
        ``str(result)`` is the markdown, so legacy callers that treated the return
        value as a string keep working.
        """
        report_config = report_config or ReportConfig()
        await self.initialize()

        # Track the last final-response text per stage. The report is the last
        # post-stage's output (the aggregator); if no post stage produced text we
        # fall back to the orchestrator's own final answer.
        stage_final: dict = {}
        post_names = [a.name for a in pipeline_post_agents]

        async for stage_name, event in self.run_pipeline(query, report_config, verbose=verbose):
            text = self._final_text(event)
            if text is not None:
                stage_final[stage_name] = text

        report_markdown = ""
        for name in reversed(post_names):
            if stage_final.get(name):
                report_markdown = stage_final[name]
                break
        if not report_markdown:
            report_markdown = stage_final.get(root_agent.name, "")

        # Read session state once, for the S3 fallback and reference extraction.
        try:
            session = await self.session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id,
            )
            state = dict(getattr(session, "state", None) or {}) if session else {}
        except Exception:
            state = {}

        # Safety net: if the aggregator produced nothing (e.g. the run stopped
        # early), surface any captured S3 artifacts so results still reach the
        # user. In the normal path the aggregator already embeds these, so we
        # only fall back when there is no report text.
        if not report_markdown.strip():
            arts = state.get("fedot_artifacts")
            if arts:
                blocks = []
                for a in arts:
                    url = a.get("url")
                    if not url:
                        continue
                    cnt = a.get("generated_count")
                    tag = f" ({cnt} molecules)" if cnt else ""
                    preview = await asyncio.to_thread(_s3_csv_preview, url)
                    block = f"**Generated result{tag}** — [download full CSV]({url})"
                    if preview:
                        block += f"\n```\n{preview}\n```"
                    blocks.append(block)
                if blocks:
                    report_markdown = "## Captured results\n\n" + "\n\n".join(blocks)

        if not report_markdown.strip():
            if self._run_error is not None:
                report_markdown = (
                    f"The run stopped early ({type(self._run_error).__name__}) before producing a "
                    "result, and no partial artifacts were captured. This is usually a slow MCP "
                    "tool hitting its timeout or a transient model/network error — please retry."
                )
            else:
                report_markdown = (
                    "I couldn't complete this request within the available steps — the orchestrator "
                    "did not reach a tool that produced a result. Please retry or narrow the request."
                )

        # Package the deliverable: report.md + LaTeX (per config) + MANIFEST.json.
        return await asyncio.to_thread(
            finalize_report, self.session_id, report_markdown, report_config, state,
        )

    async def close(self):
        """Cleanup session-related resources and uploaded paper artifacts."""
        try:
            await asyncio.to_thread(cleanup_uploaded_papers, self.user_id, self.session_id)
        except Exception as exc:
            logger.error(f"Warning: failed to cleanup uploaded papers for session {self.session_id}: {exc}")

# Convenience functions
async def create_manager() -> CoScientistManager:
    """Create and initialize a CoScientistManager."""
    manager = CoScientistManager()
    await manager.initialize()
    return manager


# Export public API
__all__ = [
    # Main classes
    "CoScientistManager",
    "ReportConfig",
    "RunResult",
    # Functions
    "create_manager",
]

# CLI entrypoint
if __name__ == "__main__":
    import argparse
    from CoScientist.config import LATEX_MODES

    async def main():
        parser = argparse.ArgumentParser(description="CoScientist interactive CLI")
        parser.add_argument(
            "--latex", choices=LATEX_MODES, default="skip",
            help="LaTeX output mode for the final report (default: skip).",
        )
        args = parser.parse_args()
        report_config = ReportConfig.from_cli(args)

        manager = await create_manager()

        print("CoScientist (ADK) initialized\n")

        try:
            while True:
                query = input("Enter query (or 'exit'): ")

                if query.lower() in {"exit", "quit"}:
                    break

                result = await manager.run(query, report_config=report_config)

                print("\n=== Final Response ===")
                print(result.markdown)
                if result.report_dir:
                    print(f"\n📁 Report deliverable: {result.report_dir}")
                print()

        finally:
            await manager.close()

    asyncio.run(main())
