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


def cleanup_plan(plan_file_path: str) -> None:

    with open(plan_file_path, "r", encoding="utf-8") as f:
        plan = f.read()

    # ensure non-empty
    if not plan or not plan.strip():
        raise ValueError("Plan cannot be empty.")

    # ensure at least 1 step
    if "1)" not in plan:
        raise ValueError("Plan must contain at least one step. Found: \n" + plan)

    # fallback protection: if it contains literally "1)", fix it
    if "1)" in plan and plan.strip() == "1)":
        raise ValueError("Plan contains only '1)'. Add a valid step.")

    if " 1)" in plan:
        plan_idx = plan.find("1)")
        plan = plan[plan_idx:]
        print("\n\n\nPlan has been cleaned up. It now starts from step 1.\n\n\n")

    if "ReporterAgent" in plan:
        report_idx = plan.find("ReporterAgent")
        plan = plan[:report_idx]
        last_idx = plan.rfind(")")
        plan = plan[:last_idx-1]
        print("\n\n\nPlan has been cleaned up. It doesn't contain ReporterAgent.\n\n\n")

    with open(plan_file_path, "w", encoding="utf-8") as f:
        f.write(plan)

    return

class SessionAgent(LlmAgent):
    """A planner that generates a roadmap and asks the human.
    If the human requests changes, it automatically feeds the changes back
    to itself and generates a new roadmap, looping until approved.
    """
    hitl_handler: Optional[AbstractHITLHandler] = None
    plan_file_path: Optional[str] = None
    correction_prompt: str = "The human reviewed your output and provided this feedback/correction:\n\n{feedback}\n\nYou MUST rewrite your output incorporating this feedback."

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

            if not self.hitl_handler:
                # No HITL: just pass the final event through and exit
                if final_event is not None:
                    yield final_event
                break

            if self.output_key:
                output_text = ctx.session.state.get(self.output_key, output_text)

            # Perform HITL check
            message = f"[INTERNAL_LOOP: SessionAgent] Agent '{self.name}' proposes its result. Please review."

            # If plan_file_path is set, write to file and update message
            if self.plan_file_path:
                try:
                    with open(self.plan_file_path, "w", encoding="utf-8") as f:
                        f.write(str(output_text))
                    message += f"\n\n--> The plan has been recorded to '{self.plan_file_path}'. You can edit it before approving."
                except Exception as e:
                    message += f"\n\n[Warning] Failed to write plan to {self.plan_file_path}: {e}"

            
            cleanup_plan(self.plan_file_path)

            request = HITLRequest(
                agent_name=self.name,
                action_type=HITLAction.APPROVE,
                message=message,
                context={"output": str(output_text)},
                invoked_via="internal_loop"
            )

            response = await self.hitl_handler.handle_request(request)

            if response.approved:
                if self.plan_file_path:
                    try:
                        if os.path.exists(self.plan_file_path):
                            with open(self.plan_file_path, "r", encoding="utf-8") as f:
                                edited_content = f.read()

                            if self.output_key:
                                ctx.session.state[self.output_key] = edited_content
                                print(f"\n[SessionAgent] SUCCESS: Updated '{self.output_key}' from '{self.plan_file_path}'.")

                                # Replace final_event content with the edited plan
                                final_event = Event(
                                    invocation_id=ctx.invocation_id,
                                    author=self.name,
                                    branch=ctx.branch,
                                    content=types.Content(
                                        role="model",
                                        parts=[types.Part(text=edited_content)]
                                    )
                                )
                    except Exception as e:
                        print(f"Error reading plan from {self.plan_file_path}: {e}")

                if not response.free_input and response.action != HITLAction.EDIT:
                    # HITL approved — now emit the (possibly updated) final event and exit
                    if final_event is not None:
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

