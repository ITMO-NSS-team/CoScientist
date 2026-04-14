import asyncio
from typing import AsyncGenerator, Optional

from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.utils.context_utils import Aclosing

from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLRequest, HITLAction

class LoopingAgent(LlmAgent):
    """A planner that generates a plan and asks the human.
    If the human requests changes, it automatically feeds the changes back
    to itself and generates a new plan, looping until approved.
    """
    hitl_handler: Optional[AbstractHITLHandler] = None
    correction_prompt: str = "The human reviewed your output and provided this feedback/correction:\n\n{feedback}\n\nYou MUST rewrite your output incorporating this feedback. If you had an output schema, you MUST still follow it strictly and return a valid JSON object."

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
            request = HITLRequest(
                agent_name=self.name,
                action_type=HITLAction.APPROVE,
                message=f"[INTERNAL_LOOP: AGENT_LOGIC] Agent '{self.name}' proposes its result. Please review.",
                context={"output": str(output_text)},
                invoked_via="internal_loop"
            )
            
            response = await self.hitl_handler.handle_request(request)
            
            # If approved and there is free input, OVERWRITE the result with user text
            if response.approved:
                if response.free_input:
                    output_text = response.free_input
                    if self.output_key:
                        ctx.session.state[self.output_key] = output_text
                
                if response.action != HITLAction.EDIT:
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

