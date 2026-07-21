import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.utils.context_utils import Aclosing

from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse

import json
from CoScientist.tools.task_tracker import task_tracker_instance

logger = logging.getLogger("CoScientist.hitl.session_agent")


def render_task_plan(tasks) -> str:
    """Render the registered delegation plan without model/tool narration."""
    if not isinstance(tasks, list) or not tasks:
        return ""
    lines = [f"План: {len(tasks)} задач(и)"]
    for index, task in enumerate(tasks, 1):
        title = task.get("title") or f"Задача {index}"
        assignee = task.get("assignee") or "не назначен"
        lines.extend((f"\n{index}. {title}", f"Исполнитель: {assignee}"))
        description = (task.get("description") or "").strip()
        if description:
            lines.append(description)
        depends_on = task.get("parent_id")
        if depends_on:
            lines.append(f"Зависит от: {depends_on}")
    return "\n".join(lines)


class SessionAgent(LlmAgent):
    """A planner that generates a roadmap and asks the human.
    If the human requests changes, it automatically feeds the changes back
    to itself and generates a new roadmap, looping until approved.
    """
    hitl_handler: Optional[AbstractHITLHandler] = None
    correction_prompt: str = "The human reviewed your output and provided this feedback/correction:\n\n{feedback}\n\nYou MUST rewrite your output incorporating this feedback."

    def _review_output(self, output_text) -> str:
        """How the proposed output is presented to the human reviewer.

        Structured outputs (dict/list from an output_schema) are shown as
        readable JSON. Subclasses may override to show a rendered document
        instead (e.g. the microfluidics ТЗ agent renders Markdown)."""
        if isinstance(output_text, (dict, list)):
            try:
                return json.dumps(output_text, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                pass
        return str(output_text)

    def _post_final_events(self, ctx: InvocationContext, output_text):
        """Extra events to emit AFTER the final output is accepted (approved
        by the human, or produced directly when no HITL handler is wired).

        Subclasses may yield follow-up events — e.g. the microfluidics ТЗ
        agent publishes the rendered ТЗ document into the chat before the
        pipeline moves on. Default: nothing."""
        return iter(())

    async def _review_decision(self, ctx: InvocationContext, output_text) -> HITLResponse:
        """One review round with the human; returns the final decision.

        Default: a single approve/edit request showing the proposed output.
        Subclasses may run a multi-step dialogue instead (e.g. the ТЗ agent
        first reviews the document, then interviews the operator question by
        question) as long as they return one HITLResponse."""
        review_output = output_text
        if self.name == "PlannerAgent":
            registered_plan = render_task_plan(ctx.session.state.get("active_tasks"))
            if registered_plan:
                review_output = registered_plan

        request = HITLRequest(
            agent_name=self.name,
            action_type=HITLAction.APPROVE,
            message=(
                f"[INTERNAL_LOOP: SessionAgent] Agent '{self.name}' proposes "
                "its result. Please review."
            ),
            context={"output": self._review_output(review_output)},
            invoked_via="internal_loop",
        )
        return await self.hitl_handler.handle_request(request)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:

        while True:
            output_text = ""
            final_event = None

            # Never review a stale plan from an earlier attempt/session turn if
            # the current planner run fails before create_plan succeeds.
            if self.name == "PlannerAgent":
                ctx.session.state.pop("active_tasks", None)

            async with Aclosing(super()._run_async_impl(ctx)) as agen:
                async for event in agen:
                    if event.is_final_response():
                        final_event = event
                        # Earlier text events contain reasoning and tool-call
                        # narration. Only the final response may reach HITL.
                        output_text = "".join(
                            part.text or ""
                            for part in (event.content.parts if event.content else [])
                        )
                    else:
                        yield event

            from CoScientist.config import get_settings
            hitl_on = bool(self.hitl_handler and get_settings().web.hitl_enabled)

            if not hitl_on or final_event is None:
                # No HITL or not a final event (e.g. tool call): just pass and exit
                if not hitl_on:
                    logger.info(
                        "%s: HITL disabled (hitl_enabled=False) — "
                        "output passed through without human review", self.name,
                    )
                if final_event is not None:
                    yield final_event
                    for extra in self._post_final_events(ctx, output_text):
                        yield extra
                break

            if self.output_key:
                output_text = ctx.session.state.get(self.output_key, output_text)

            # Perform HITL check (subclasses may run a multi-step dialogue).
            response = await self._review_decision(ctx, output_text)

            if response.approved:
                if response.instructions and response.action != HITLAction.EDIT:
                    edited_text = response.instructions
                    if final_event is not None and final_event.content and final_event.content.parts:
                        final_event.content.parts[0].text = edited_text
                    if self.output_key:
                        ctx.session.state[self.output_key] = edited_text

                    try:
                        parsed = json.loads(edited_text)
                        if isinstance(parsed, list):
                            class DummyContext:
                                def __init__(self, state):
                                    self.state = state
                            task_tracker_instance.create_plan(parsed, DummyContext(ctx.session.state))
                    except Exception:
                        pass

                if not response.free_input and response.action != HITLAction.EDIT:
                    # HITL approved — now emit the (possibly updated) final event and exit
                    if final_event is not None:
                        yield final_event
                        for extra in self._post_final_events(ctx, output_text):
                            yield extra
                    break

            # Rejected or "Edit" requested — feed feedback back into the agent
            feedback = response.instructions or response.free_input or "No feedback provided."

            user_feedback_event = Event(
                invocation_id=ctx.invocation_id,
                author="user",
                branch=ctx.branch,
                content=types.Content(
                    role="user",
                    parts=[types.Part(text=self.correction_prompt.format(feedback=feedback))]
                )
            )

            ctx.session.events.append(user_feedback_event)
            yield user_feedback_event

            # Clear end-of-agent flag so the agent is allowed to re-run
            ctx.set_agent_state(self.name)

