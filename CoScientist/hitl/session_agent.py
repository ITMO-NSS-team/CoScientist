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
from CoScientist.hitl.models import HITLRequest, HITLAction

import json
from CoScientist.tools.task_tracker import task_tracker_instance

logger = logging.getLogger("CoScientist.hitl.session_agent")

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

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:

        while True:
            output_text = ""
            final_event = None

            async with Aclosing(super()._run_async_impl(ctx)) as agen:
                async for event in agen:
                    # Collect text for potential HITL refinement
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                output_text += part.text

                    if event.is_final_response():
                        # Hold — emit only after HITL decision
                        final_event = event
                    else:
                        yield event

            if not self.hitl_handler or final_event is None:
                # No HITL or not a final event (e.g. tool call): just pass and exit
                if not self.hitl_handler:
                    logger.info(
                        "%s: no HITL handler wired (HITL__ENABLED off?) — "
                        "output passed through without human review", self.name,
                    )
                if final_event is not None:
                    yield final_event
                    for extra in self._post_final_events(ctx, output_text):
                        yield extra
                break

            if self.output_key:
                output_text = ctx.session.state.get(self.output_key, output_text)

            # Perform HITL check
            message = f"[INTERNAL_LOOP: SessionAgent] Agent '{self.name}' proposes its result. Please review."

            request = HITLRequest(
                agent_name=self.name,
                action_type=HITLAction.APPROVE,
                message=message,
                context={"output": self._review_output(output_text)},
                invoked_via="internal_loop"
            )

            response = await self.hitl_handler.handle_request(request)

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

