import asyncio
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

class SessionAgent(LlmAgent):
    """A planner that generates a roadmap and asks the human.
    If the human requests changes, it automatically feeds the changes back
    to itself and generates a new roadmap, looping until approved.
    """
    hitl_handler: Optional[AbstractHITLHandler] = None
    result_state_key: Optional[str] = None
    correction_prompt: str = "The human reviewed your output and provided this feedback/correction:\n\n{feedback}\n\nYou MUST rewrite your output incorporating this feedback."

    @staticmethod
    def _fallback_plan(ctx: InvocationContext) -> list[dict]:
        """Create a minimal executable plan when the planner skipped its tool.

        Tool calling is probabilistic even when the instruction explicitly requires
        ``create_plan``.  The orchestration layer, however, requires durable tasks
        to continue.  This fallback is deliberately narrow: it is used only for
        the PlannerAgent's ``active_tasks`` contract and is persisted through the
        normal task tracker just like a model-generated plan.
        """
        query = "the user's request"
        for event in reversed(getattr(ctx.session, "events", ())):
            content = getattr(event, "content", None)
            if not content or getattr(content, "role", None) != "user":
                continue
            for part in getattr(content, "parts", ()):
                text = getattr(part, "text", None)
                if text:
                    query = text.strip()
                    break
            if query != "the user's request":
                break

        return [{
            "title": "Investigate the user request",
            "description": f"Investigate and answer: {query}",
            "assignee": "ResearchAgent",
            "notes": (
                "Automatically created because PlannerAgent completed without "
                "calling create_plan."
            ),
        }]

    def _ensure_result_state(self, ctx: InvocationContext) -> None:
        """Ensure the planner has a serializable, executable result state."""
        if self.result_state_key != "active_tasks":
            return
        if self.result_state_key in ctx.session.state:
            return

        class _StateContext:
            def __init__(self, state):
                self.state = state

        outcome = task_tracker_instance.create_plan(
            self._fallback_plan(ctx), _StateContext(ctx.session.state)
        )
        if outcome.get("result") != "success":
            raise RuntimeError(
                f"SessionAgent '{self.name}' could not create fallback plan: "
                f"{outcome.get('message', 'unknown error')}"
            )

    def _apply_state_result(
        self,
        ctx: InvocationContext,
        final_event: Event,
    ) -> None:
        """Replace the final response with deterministic JSON from session state."""
        if not self.result_state_key:
            return

        state = ctx.session.state
        if self.result_state_key not in state:
            self._ensure_result_state(ctx)
        if self.result_state_key not in state:
            raise RuntimeError(
                f"SessionAgent '{self.name}' cannot return its result: "
                f"session state key '{self.result_state_key}' is missing."
            )

        result_text = json.dumps(
            state[self.result_state_key],
            ensure_ascii=False,
            indent=2,
        )

        if final_event.content is None:
            final_event.content = types.Content(
                role="model",
                parts=[types.Part(text=result_text)],
            )
        elif not final_event.content.parts:
            final_event.content.parts = [types.Part(text=result_text)]
        else:
            for part in final_event.content.parts:
                if part.text is not None:
                    part.text = result_text
                    break
            else:
                final_event.content.parts.insert(0, types.Part(text=result_text))

        if self.output_key:
            state[self.output_key] = result_text

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
                if final_event is not None:
                    self._apply_state_result(ctx, final_event)
                    yield final_event
                break

            if self.output_key:
                output_text = ctx.session.state.get(self.output_key, output_text)

            # Perform HITL check
            message = f"[INTERNAL_LOOP: SessionAgent] Agent '{self.name}' proposes its result. Please review."

            request = HITLRequest(
                agent_name=self.name,
                action_type=HITLAction.APPROVE,
                message=message,
                context={"output": str(output_text)},
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
                        self._apply_state_result(ctx, final_event)
                        yield final_event
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

