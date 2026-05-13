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
            
            # Delegate to normal LlmAgent generation
            async with Aclosing(super()._run_async_impl(ctx)) as agen:
                async for event in agen:
                    # Collect text for potential HITL refinement
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                output_text += part.text
                    
                    final_event = event
                    yield event

            if not self.hitl_handler:
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

            request = HITLRequest(
                agent_name=self.name,
                action_type=HITLAction.APPROVE,
                message=message,
                context={"output": str(output_text)},
                invoked_via="internal_loop"
            )
            
            response = await self.hitl_handler.handle_request(request)
            
            # If approved without any further instructions, we are done.
            # But first, check if we should read back the edited plan from the file
            if response.approved:
                if self.plan_file_path:
                    try:
                        if os.path.exists(self.plan_file_path):
                            with open(self.plan_file_path, "r", encoding="utf-8") as f:
                                edited_content = f.read()
                            
                            # If edited_content is different, we use it as the final output
                            if self.output_key:
                                ctx.session.state[self.output_key] = edited_content
                                print(f"\n[SessionAgent] SUCCESS: Updated '{self.output_key}' from '{self.plan_file_path}'.")
                                
                                # Yield an informative event that content was updated from file
                                yield Event(
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
                    break
            
            # If rejected or "Edit" requested
            feedback = response.instructions or response.free_input or "No feedback provided."

            # Yield an event that represents the user's feedback natively into the history!
            user_feedback_event = Event(
                invocation_id=ctx.invocation_id,
                author="user",
                branch=ctx.branch,
                content=types.Content(
                    role="user",
                    parts=[types.Part(text=self.correction_prompt.format(feedback=feedback))]
                )
            )
            
            # Add to session state so the LLM flow sees it for the next loop
            ctx.session.events.append(user_feedback_event)
            yield user_feedback_event
            
            # Clear end_of_agent flag so the agent is allowed to re-run
            ctx.set_agent_state(self.name)

