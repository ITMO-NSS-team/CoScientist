"""
HITL Callbacks — Variant B (commented out).

This module provides an alternative HITL approach using ADK agent callbacks.
Instead of the agent explicitly calling a tool, callbacks automatically
intercept agent output and request human confirmation.

Uncomment and use if you prefer system-level HITL control
(the system decides when to ask, not the agent).
"""

# import re
# from typing import Optional
#
# from google.genai import types as genai_types
#
# from CoScientist.hitl.models import HITLRequest, HITLAction
# from CoScientist.hitl.handler import AbstractHITLHandler
#
#
# def _parse_options(text: str) -> list[str]:
#     """Extract numbered options from agent output text."""
#     return re.findall(r'^\d+[.)]\s*(.+)$', text, re.MULTILINE)
#
#
# def make_hitl_after_callback(handler: AbstractHITLHandler, action_type: HITLAction):
#     """Factory for after_agent_callback that intercepts agent output and requests HITL.
#
#     Usage:
#         hypotheses_agent = LlmAgent(
#             name="HypothesesAgent",
#             ...
#             after_agent_callback=make_hitl_after_callback(handler, HITLAction.SELECT),
#         )
#
#     Args:
#         handler: HITL handler instance (Console, Callback, etc.)
#         action_type: Type of HITL action to request (APPROVE, SELECT, etc.)
#
#     Returns:
#         An async callback function compatible with ADK's after_agent_callback.
#     """
#
#     async def after_agent_callback(callback_context) -> Optional[genai_types.Content]:
#         agent_name = callback_context.agent_name
#         state = callback_context.state
#
#         # Get agent output from session state
#         output_key = callback_context.agent.output_key
#         agent_output = state.get(output_key, "")
#
#         if not agent_output:
#             return None  # No output to review
#
#         request = HITLRequest(
#             agent_name=agent_name,
#             action_type=action_type,
#             message=f"Agent '{agent_name}' proposes:\n{agent_output}",
#             options=_parse_options(str(agent_output)) if action_type == HITLAction.SELECT else [],
#         )
#
#         response = await handler.handle_request(request)
#
#         if not response.approved:
#             # Override agent output with rejection feedback
#             return genai_types.Content(
#                 role="model",
#                 parts=[genai_types.Part(
#                     text=f"Human rejected the proposal. Feedback: {response.free_input or 'No feedback provided'}"
#                 )],
#             )
#
#         if response.selected_option:
#             # Override agent output with human's selection
#             return genai_types.Content(
#                 role="model",
#                 parts=[genai_types.Part(
#                     text=f"Human selected: {response.selected_option}"
#                 )],
#             )
#
#         # None = agent output is accepted as-is
#         return None
#
#     return after_agent_callback
#
#
# def make_hitl_before_callback(handler: AbstractHITLHandler):
#     """Factory for before_agent_callback that asks for confirmation before agent runs.
#
#     Usage:
#         experiment_agent = LlmAgent(
#             name="ExperimentAgent",
#             ...
#             before_agent_callback=make_hitl_before_callback(handler),
#         )
#     """
#
#     async def before_agent_callback(callback_context) -> Optional[genai_types.Content]:
#         agent_name = callback_context.agent_name
#
#         request = HITLRequest(
#             agent_name=agent_name,
#             action_type=HITLAction.APPROVE,
#             message=f"Agent '{agent_name}' is about to execute. Approve?",
#         )
#
#         response = await handler.handle_request(request)
#
#         if not response.approved:
#             return genai_types.Content(
#                 role="model",
#                 parts=[genai_types.Part(
#                     text=f"Human declined to run agent '{agent_name}'. Reason: {response.free_input or 'No reason given'}"
#                 )],
#             )
#
#         return None  # Proceed normally
#
#     return before_agent_callback
