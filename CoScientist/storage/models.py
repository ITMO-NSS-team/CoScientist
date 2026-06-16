from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class RetrievalFinalResult(BaseModel):
    """Result from a retrieval query."""
    servers_id: List[str] = Field(default_factory=list, description="List of unique identifier of selected MCP servers")
    queries: List[str] = Field(default_factory=list, description="Queries used to retrieve MCP servers")
    task: str = Field(..., description="Original task")

class RetrievalToolResult(BaseModel):
    tool: str
    server_id: str
    description: str
    score: float

class ToolScore(BaseModel):
    index: int
    score: float

class ToolRanking(BaseModel):
    tools: List[ToolScore]


class MCPScore(BaseModel):
    index: int
    score: bool

class MCPRanking(BaseModel):
    mcp_scores: List[MCPScore]
    reasoning: str