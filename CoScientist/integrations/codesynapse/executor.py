"""Adapter that runs the existing full CoScientist manager for the façade."""

from __future__ import annotations

from typing import Protocol

from CoScientist.hitl.handler import AbstractHITLHandler


class PipelineExecutor(Protocol):
    async def execute(self, request: object, hitl_handler: AbstractHITLHandler) -> str: ...


class ManagerPipelineExecutor:
    """Production executor retaining the existing full-manager execution path."""

    async def execute(self, request: object, hitl_handler: AbstractHITLHandler) -> str:
        # Delayed import keeps contract tests independent of model/MCP setup.
        from CoScientist.main import CoScientistManager

        plugins = []
        trace_recorder = getattr(request, "trace_recorder", None)
        if trace_recorder is not None:
            from CoScientist.integrations.codesynapse.trace_plugin import CodesynapseTracePlugin

            plugins.append(CodesynapseTracePlugin(trace_recorder))
        manager = CoScientistManager(
            app_name=f"codesynapse-{request.coscientist_run_id}",
            user_id=request.tenant_id,
            session_id=request.coscientist_run_id,
            hitl_handler=hitl_handler,
            plugins=plugins,
        )
        try:
            return await manager.run(request.research_request, verbose=False)
        finally:
            await manager.close()
