"""TaskExecutionAgent pipeline.

Sequential pipeline:
  ToolPreparerAgent (sequential)
    ├── ParallelToolSearcherAgent
    │     ├── LocalToolsExtractorAgent  (RAG retrieval + reranking)
    │     └── ToolWebSearcherAgent      (web MCP discovery)
    ├── FullSetToolReranker             (score web MCPs vs local)
    └── WebToolsDeployerAgent           (deploy chosen MCPs)
  ExperimentAgent                       (run FEDOT.MAS)
"""
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent

from CoScientist.agents.callbacks import (
    after_fullset_reranker_agent,
    after_tool_reranker_agent,
    before_tool_reranker_model,
)
from CoScientist.agents.common import agent_tools, make_llm
from CoScientist.agents.custom_agents import WebToolsDeployerAgent
from CoScientist.agents.prompts import (
    fedot_instruction,
    tool_reranker_instruction,
    tool_retriever_instruction,
    tool_scoring_instruction,
    tool_websearcher_instruction,
)
from CoScientist.storage import MCPRanking, ToolRanking
from CoScientist.tools import (
    fedot_toolset_instance,
    retrieval_toolset_instance,
    search_mcp_servers,
)

# ── Local RAG retrieval ──────────────────────────────────────────────────────

tool_retriever_agent = LlmAgent(
    name="ToolRetrieverAgent",
    model=make_llm(),
    instruction=tool_retriever_instruction,
    description="Retrieve relevant MCP servers from the RAG database for a given task.",
    tools=retrieval_toolset_instance,
    output_key="retrieved_tools",
)

tool_reranker_agent = LlmAgent(
    name="ToolReranker",
    model=make_llm(),
    instruction=tool_reranker_instruction,
    description="Rerank retrieved MCP servers by relevance for a given task.",
    output_schema=ToolRanking,
    before_model_callback=before_tool_reranker_model,
    after_agent_callback=after_tool_reranker_agent,
    output_key="reranked_tools",
)

local_tools_extractor = SequentialAgent(
    name="LocalToolsExtractorAgent",
    sub_agents=[tool_retriever_agent, tool_reranker_agent],
    description="Extract relevant ready-to-use tools from local storage",
)

# ── Web MCP discovery ────────────────────────────────────────────────────────

tool_websearcher_agent = LlmAgent(
    name="ToolWebSearcherAgent",
    model=make_llm(),
    instruction=tool_websearcher_instruction,
    description="Web-search for relevant MCP servers in public repositories.",
    tools=[search_mcp_servers],
    output_key="retrieved_web_tools",
)

tool_searcher = ParallelAgent(
    name="ParallelToolSearcherAgent",
    sub_agents=[local_tools_extractor, tool_websearcher_agent],
    description="Run local RAG retrieval and web MCP search in parallel.",
)

# ── Cross-source reranking & deployment ─────────────────────────────────────

tool_fullset_reranker_agent = LlmAgent(
    name="FullSetToolReranker",
    model=make_llm(),
    instruction=tool_scoring_instruction,
    description="Score web-found MCP servers against already available local ones.",
    output_schema=MCPRanking,
    after_agent_callback=after_fullset_reranker_agent,
    output_key="reranked_web_servers",
)

web_tools_deployer = WebToolsDeployerAgent(
    name="WebToolsDeployerAgent",
    description="Deploy the selected web MCP servers",
)

tool_agent = SequentialAgent(
    name="ToolPreparerAgent",
    sub_agents=[tool_searcher, tool_fullset_reranker_agent, web_tools_deployer],
    description="Find and prepare relevant MCP servers for the current task",
)

# ── Experiment runner ────────────────────────────────────────────────────────

fedot_agent = LlmAgent(
    name="ExperimentAgent",
    model=make_llm(),
    instruction=fedot_instruction,
    description="Invoke the FEDOT multi-agent system to solve the task using MCP tools.",
    output_key="fedot_results",
    tools=agent_tools(fedot_toolset_instance, hitl=True),
)

# ── Top-level sequential agent ───────────────────────────────────────────────

task_execution_agent = SequentialAgent(
    name="TaskExecutorAgent",
    sub_agents=[tool_agent, fedot_agent],
    description=(
        "Complete experiments and run calculations. "
        "Discovers and deploys MCP tools, then runs the experiment pipeline."
    ),
)
