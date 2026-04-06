"""Data models for Human-in-the-Loop (HITL) module."""

from enum import Enum
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field


class HITLAction(str, Enum):
    """Types of HITL interactions."""
    APPROVE = "approve"
    REJECT = "reject"
    SELECT = "select"
    EDIT = "edit"
    PROVIDE_INPUT = "provide_input"


class HITLRequest(BaseModel):
    """Request from an agent to a human."""
    agent_name: str = Field(..., description="Name of the agent making the request")
    action_type: HITLAction = Field(..., description="Type of action requested")
    message: str = Field(..., description="Message to the human")
    options: List[str] = Field(default_factory=list, description="Options for selection")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timeout_seconds: Optional[float] = Field(default=None, description="Timeout for the request")


class HITLResponse(BaseModel):
    """Response from a human to an agent."""
    action: HITLAction = Field(..., description="Action taken by the human")
    selected_option: Optional[str] = Field(default=None, description="Selected option (for SELECT)")
    edited_content: Optional[str] = Field(default=None, description="Edited content (for EDIT)")
    free_input: Optional[str] = Field(default=None, description="Free-form input")
    approved: bool = Field(default=False, description="Whether the action was approved")
