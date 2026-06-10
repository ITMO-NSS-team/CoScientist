"""Agent callbacks: tool/reranker, critic, medical, and research callbacks.

Re-exported here so callers can use `from CoScientist.agents.callbacks import
<name>` regardless of which submodule defines it.
"""
from CoScientist.agents.callbacks.critic import (
    post_action_critique,
    pre_action_critique,
)
from CoScientist.agents.callbacks.med_callbacks import (
    before_model_modifier,
    med_agent_before_model,
)
from CoScientist.agents.callbacks.research_callbacks import (
    cleanup_uploaded_papers,
    ensure_local_papers_uploaded,
    papers_agent_before_model,
)
from CoScientist.agents.callbacks.tool_callbacks import (
    after_fullset_reranker_agent,
    after_tool_reranker_agent,
    before_tool_reranker_model,
    print_research_agent_tool_call,
)

__all__ = [
    "pre_action_critique",
    "post_action_critique",
    "before_model_modifier",
    "med_agent_before_model",
    "papers_agent_before_model",
    "ensure_local_papers_uploaded",
    "cleanup_uploaded_papers",
    "before_tool_reranker_model",
    "after_tool_reranker_agent",
    "after_fullset_reranker_agent",
    "print_research_agent_tool_call",
]
