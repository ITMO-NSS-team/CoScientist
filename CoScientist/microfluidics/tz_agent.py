"""ТЗ session agent + document publication for the microfluidics profile.

``TZSessionAgent`` is the SessionAgent used for TZSpecAgent:

  * during the HITL review loop the human is shown the RENDERED ТЗ document
    (the reference Markdown format), not the raw JSON the schema needs;
  * once the ТЗ is accepted (approved by the human, or produced directly in
    headless mode) the agent PUBLISHES the document: saves it to
    ``tz_documents/TZ_<timestamp>.md``, stores it in session state under
    ``structured_tz_document`` and emits a chat message with the file path,
    the web link (/api/tz-document) and the full document text — right before
    the pipeline moves on to planning / literature search.

``save_tz_document`` remains available as an after_agent callback for
profiles that want persistence without the chat message.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from CoScientist.hitl.session_agent import SessionAgent
from CoScientist.microfluidics.render import render_tz_document

logger = logging.getLogger(__name__)

TZ_DOCUMENT_STATE_KEY = "structured_tz_document"
TZ_DOCUMENTS_DIR = Path("tz_documents")


def persist_tz_document(tz, out_dir: Optional[Path] = None) -> Tuple[str, Optional[Path]]:
    """Render the ТЗ document and write it to disk.

    Returns ``(document_markdown, saved_path)``; ``saved_path`` is None when
    the file could not be written (the document itself is still returned).
    Raises only if the ТЗ cannot be rendered at all.
    """
    document = render_tz_document(tz)
    out_dir = out_dir or TZ_DOCUMENTS_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"TZ_{stamp}.md"
        path.write_text(document, encoding="utf-8")
        logger.info("ТЗ document saved to %s", path)
        return document, path
    except OSError as exc:
        logger.warning("Could not write the ТЗ document file: %s", exc)
        return document, None


class TZSessionAgent(SessionAgent):
    """SessionAgent that reviews and publishes the rendered ТЗ document.

    The review shows the rendered document PLUS the operator questionnaire
    (ported from VibePAV) for every block that still has «не задано» fields —
    the operator may answer any subset (`Qn: <ответ>`), say «не знаю» to keep
    a field open, or delegate with «на усмотрение агента»."""

    def _review_output(self, output_text) -> str:
        from CoScientist.microfluidics.questionnaire import (
            build_questionnaire,
            render_questionnaire,
        )
        try:
            document = render_tz_document(output_text)
        except Exception as exc:  # noqa: BLE001 — review must never crash the run
            logger.warning("TZ review render failed (%s); showing raw output", exc)
            return str(output_text)
        try:
            questions = build_questionnaire(output_text)
            if questions:
                document += "\n" + render_questionnaire(questions)
        except Exception as exc:  # noqa: BLE001 — the questionnaire is an aid only
            logger.warning("TZ questionnaire build failed: %s", exc)
        return document

    def _post_final_events(self, ctx: InvocationContext, output_text):
        """Publish the accepted ТЗ document into the chat (file + state + text)."""
        tz = None
        if self.output_key:
            tz = ctx.session.state.get(self.output_key)
        if not tz:
            tz = output_text
        try:
            document, path = persist_tz_document(tz)
        except Exception as exc:  # noqa: BLE001 — publishing must not kill the run
            logger.warning("TZ document publication failed: %s", exc)
            return

        header = "📄 Структурированное ТЗ сформировано."
        if path is not None:
            header += f"\nФайл: {path.resolve()}"
        header += "\nОткрыть в браузере: /api/tz-document"
        text = f"{header}\n\n{document}"

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            actions=EventActions(state_delta={TZ_DOCUMENT_STATE_KEY: document}),
        )


def save_tz_document(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """after_agent callback: persist the approved ТЗ as a Markdown document
    (state + file) WITHOUT emitting a chat message."""
    tz = callback_context.state.get("structured_tz")
    if not tz:
        return None
    try:
        document, _path = persist_tz_document(tz)
    except Exception as exc:  # noqa: BLE001 — a render bug must not kill the run
        logger.warning("TZ document render failed: %s", exc)
        return None
    callback_context.state[TZ_DOCUMENT_STATE_KEY] = document
    return None


__all__ = [
    "TZSessionAgent",
    "TZ_DOCUMENT_STATE_KEY",
    "persist_tz_document",
    "save_tz_document",
]
