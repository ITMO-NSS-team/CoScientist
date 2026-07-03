"""
HypothesisToolRegistry — dynamic registration and selection of hypothesis tools.

Starts with MooseChemTool pre-registered. Additional tools can be registered
at runtime via register() without changing any core agent code.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from CoScientist.hypothesis_subsystem.base_tool import BaseHypothesisTool
from CoScientist.hypothesis_subsystem.models import HypothesisQuery, ToolResult


class HypothesisToolRegistry:
    """
    Registry of hypothesis-generation strategy tools.

    Usage:
        registry = HypothesisToolRegistry()
        registry.register(MooseChemTool(model="gpt-4"))
        result = await registry.invoke_with_fallback(query, preferred="MooseChem")
    """

    def __init__(self, tools: Optional[List[BaseHypothesisTool]] = None):
        self._tools: Dict[str, BaseHypothesisTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, tool: BaseHypothesisTool) -> None:
        """
        Register a hypothesis-generation tool.

        Args:
            tool: A BaseHypothesisTool subclass instance.

        Raises:
            ValueError: If a tool with the same strategy_type is already registered
                        (call unregister first to replace).
        """
        if tool.strategy_type in self._tools:
            raise ValueError(
                f"Tool '{tool.strategy_type}' is already registered. "
                f"Unregister it first."
            )
        self._tools[tool.strategy_type] = tool

    def unregister(self, strategy_type: str) -> None:
        """
        Remove a tool from the registry by strategy type.

        Args:
            strategy_type: The strategy identifier to remove.
        """
        self._tools.pop(strategy_type, None)

    def get(self, strategy_type: str) -> Optional[BaseHypothesisTool]:
        """
        Get a tool by strategy type.

        Args:
            strategy_type: The strategy identifier.

        Returns:
            The tool instance, or None if not registered.
        """
        return self._tools.get(strategy_type)

    def list_strategies(self) -> List[str]:
        """
        Return all registered strategy identifiers.
        """
        return list(self._tools.keys())

    def describe_all(self) -> Dict[str, str]:
        """
        Return a dict of strategy_type → description for all registered tools.
        """
        return {st: tool.describe() for st, tool in self._tools.items()}

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def invoke_with_fallback(
        self,
        query: HypothesisQuery,
        preferred: Optional[str] = None,
    ) -> ToolResult:
        """
        Invoke the preferred tool, falling back to first available if missing.

        Args:
            query: The hypothesis generation query.
            preferred: Preferred strategy type. If None or not found, uses the
                       first registered tool.

        Returns:
            ToolResult from the invoked strategy.

        Raises:
            RuntimeError: If no tools are registered.
        """
        if not self._tools:
            raise RuntimeError("No hypothesis tools registered in the registry.")

        tool = None
        if preferred:
            tool = self._tools.get(preferred)

        if tool is None:
            # Fall back to first available
            first_key = next(iter(self._tools))
            tool = self._tools[first_key]

        if not tool.validate_query(query):
            return ToolResult(
                strategy_type=tool.strategy_type,
                hypotheses=[],
                metadata={"reason": "Query validation failed"},
                success=False,
                error_message=f"Query is not valid for tool '{tool.strategy_type}'.",
            )

        return await tool.invoke(query)

    async def invoke_all(
        self,
        query: HypothesisQuery,
    ) -> Dict[str, ToolResult]:
        """
        Invoke ALL registered tools and return results keyed by strategy_type.

        Args:
            query: The hypothesis generation query.

        Returns:
            Dict mapping strategy_type to ToolResult.
        """
        results: Dict[str, ToolResult] = {}
        for st, tool in self._tools.items():
            if tool.validate_query(query):
                results[st] = await tool.invoke(query)
            else:
                results[st] = ToolResult(
                    strategy_type=st,
                    hypotheses=[],
                    metadata={"reason": "Query validation failed"},
                    success=False,
                    error_message=f"Query is not valid for tool '{st}'.",
                )
        return results
