"""Live tests of the HITL review loop on the microfluidics ТЗ stage.

Runs ONLY the TZAgent composite (TZSpecAgent -> TZQueryGenAgent) with the real
LLM and verifies both sides of the human-in-the-loop switch:

  1. review loop ON: a scripted "human" first sends a correction to the ТЗ
     table (EDIT) and approves the rewrite — the agent must re-generate and the
     pipeline must end with a valid ТЗ + queries in state;
  2. review loop OFF (the headless/testing mode): with no handler wired —
     exactly what HITL__ENABLED=false produces — the same stage passes through
     with no human interaction.

NOTE: this module declares HITL__ENABLED=true at import, while
test_microfluidics_e2e.py declares it false — run the integration files in
SEPARATE pytest invocations (settings are resolved once per process).

Run from the repo root:
    pytest tests/integration/test_microfluidics_hitl.py -q -s
"""
import asyncio
import contextlib
import os
import re
import sys

os.environ["HITL__ENABLED"] = "true"

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from CoScientist.assembly import build_system  # noqa: E402
from CoScientist.assembly.schema import resolve_config_path  # noqa: E402
from CoScientist.hitl.handler import AbstractHITLHandler  # noqa: E402
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(errors="replace")

QUERY = (
    "Нужен отечественный ПАВ для повышения нефтеотдачи: минерализованная вода "
    "с Ca2+/Mg2+, 60–90 °C, низкое межфазное натяжение, синтез на проточной "
    "микрофлюидной установке."
)

TZ_FEEDBACK = (
    "Уточнение оператора: масштаб результата — лабораторная рецептура и "
    "опытная партия; доступная аналитика — тензиометрия и ККМ по проводимости. "
    "Обнови соответствующие блоки ТЗ со статусом «уточнено оператором»."
)

# Answers keyed by questionnaire block: a value, an explicit «не знаю» (the
# field must stay open) and a delegation to the agent.
_Q_ANSWERS = (
    ("Масштаб результата", "лабораторная рецептура и опытная партия"),
    ("Аналитические методы", "тензиометрия и ККМ по проводимости"),
    ("Ограничения по поставкам", "не знаю"),
    ("Ограничения по себестоимости", "на усмотрение агента"),
)


def _questionnaire_feedback(proposal: str) -> str:
    """Answer the generated questionnaire by its own Qn numbering."""
    answers = []
    for block, answer in _Q_ANSWERS:
        m = re.search(r"\*\*Q(\d+) · " + re.escape(block), proposal)
        if m:
            answers.append(f"Q{m.group(1)}: {answer}")
    if not answers:  # блоки оказались заполнены черновиком — обычные правки
        return TZ_FEEDBACK
    return (
        "Ответы на опросник: " + "; ".join(answers)
        + ". Остальные вопросы: не знаю, продолжайте."
    )


class ScriptedHuman(AbstractHITLHandler):
    """Deterministic reviewer: answers the ТЗ questionnaire once, APPROVE otherwise."""

    def __init__(self):
        self.requests: list[HITLRequest] = []
        self.proposals: dict[str, list[str]] = {}
        self.sent_feedback: str = ""

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        self.requests.append(request)
        proposals = self.proposals.setdefault(request.agent_name, [])
        proposals.append(str(request.context.get("output", "")))
        if request.agent_name == "TZSpecAgent" and len(proposals) == 1:
            self.sent_feedback = _questionnaire_feedback(proposals[0])
            return HITLResponse(
                action=HITLAction.EDIT, approved=False,
                instructions=self.sent_feedback,
            )
        return HITLResponse(action=HITLAction.APPROVE, approved=True)


async def _run_tz_stage(wire_handler):
    """Build the profile, let the caller (re)wire HITL, run ONLY TZAgent.

    Returns the final session state and every text chunk the stage emitted
    into the event stream (what the chat would display)."""
    system = build_system(config_path=resolve_config_path("microfluidics"))
    wire_handler(system)

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="mf_hitl", user_id="u", session_id="s"
    )
    runner = Runner(
        agent=system.agent("TZAgent"), app_name="mf_hitl",
        session_service=session_service,
    )
    chat_texts: list[str] = []
    async for event in runner.run_async(
        user_id="u", session_id="s",
        new_message=types.Content(role="user", parts=[types.Part(text=QUERY)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None) and not getattr(part, "thought", False):
                    chat_texts.append(part.text)
    session = await session_service.get_session(
        app_name="mf_hitl", user_id="u", session_id="s"
    )
    return dict(session.state), chat_texts


def _assert_document_published_to_chat(chat_texts):
    published = [
        t for t in chat_texts
        if "Структурированное ТЗ сформировано" in t
        and "## Правила интерпретации полей" in t
    ]
    assert published, "the ТЗ document was not published into the chat stream"
    assert "tz_documents" in published[0], "chat message lacks the saved file path"
    return published[0]


def _assert_tz_artifacts(state):
    tz = state.get("structured_tz")
    assert isinstance(tz, dict) and tz.get("original_request"), f"bad ТЗ: {tz!r}"
    assert tz.get("blocks"), f"ТЗ has no blocks: {tz!r}"
    doc = state.get("structured_tz_document") or ""
    assert "## Правила интерпретации полей" in doc, "ТЗ document not rendered"
    queries = (state.get("tz_literature_queries") or {}).get("queries") or []
    assert queries and all(q.get("id") and q.get("query_en") for q in queries), (
        f"bad queries: {queries!r}"
    )
    return tz, queries


def _block(tz: dict, title: str) -> dict:
    for b in tz.get("blocks") or []:
        if str(b.get("title", "")).strip().lower() == title.lower():
            return b
    return {}


def test_review_loop_edit_then_approve():
    """The ТЗ table goes to the human; an EDIT makes the agent rewrite it."""
    human = ScriptedHuman()

    def wire(system):
        for name in ("TZSpecAgent", "TZQueryGenAgent"):
            handler = system.agent(name).hitl_handler
            assert handler is not None, f"{name}: HITL handler not wired at build"
            handler.set_delegate(human)

    state, chat_texts = asyncio.run(_run_tz_stage(wire))

    tz_proposals = human.proposals.get("TZSpecAgent", [])
    assert len(tz_proposals) == 2, (
        f"expected ТЗ proposal + rewrite after EDIT, got {len(tz_proposals)}"
    )
    assert tz_proposals[0] != tz_proposals[1], "rewrite identical to the original"
    # The reviewer must see the RENDERED document + the operator questionnaire.
    assert "## Правила интерпретации полей" in tz_proposals[0], (
        "review did not show the rendered ТЗ document"
    )
    assert "## Уточняющие вопросы оператору" in tz_proposals[0], (
        "review did not include the questionnaire"
    )
    assert human.proposals.get("TZQueryGenAgent"), "queries never went to review"
    print(f"\n[hitl] questionnaire answers sent: {human.sent_feedback}")

    tz, queries = _assert_tz_artifacts(state)
    published = _assert_document_published_to_chat(chat_texts)
    print(f"\n[hitl] review loop: {len(human.requests)} requests "
          f"({[r.agent_name for r in human.requests]}), "
          f"{len(queries)} queries after approval")
    print(f"[hitl] document published to chat "
          f"({len(published)} chars): {published.splitlines()[1]}")
    for title in ("Масштаб результата", "Аналитические методы"):
        block = _block(tz, title)
        fields = [(f.get("name"), f.get("value"), f.get("status"))
                  for f in block.get("fields") or []]
        print(f"[hitl] ТЗ «{title}» after operator feedback: {fields}")


def test_pass_through_without_handler():
    """No handler (what HITL__ENABLED=false wires) — the stage runs headless."""
    human = ScriptedHuman()

    def wire(system):
        for name in ("TZSpecAgent", "TZQueryGenAgent"):
            system.agent(name).hitl_handler = None

    state, chat_texts = asyncio.run(_run_tz_stage(wire))

    assert not human.requests, "no human should have been consulted"
    _, queries = _assert_tz_artifacts(state)
    # The document is published into the chat in headless mode too.
    _assert_document_published_to_chat(chat_texts)
    print(f"\n[hitl] pass-through: ТЗ + {len(queries)} queries + published document "
          "with zero human interaction")
