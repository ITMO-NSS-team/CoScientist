"""Minimal CoScientist workarounds for fragile FEDOT.MAS ``transfer_to_agent`` routing.

Kept separate from ``fedotmas_tools.py`` so the tool wiring stays readable and
patches survive review / can be dropped when upstream fixes land.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from google.adk.agents.base_agent import BaseAgent

from fedotmas import MAS
from fedotmas.common.llm import _ProxyClient
from fedotmas.mas.builder import _build_routing_agent, build_routing_system
from fedotmas.mas.models import MASAgentConfig, MASConfig

from CoScientist.agents.callbacks.json_output import _unwrap_completion_state

_log = logging.getLogger(__name__)

# LiteLLM-style provider prefixes. FEDOT's ``_ProxyClient`` forwards model names
# as-is to OPENAI_BASE_URL; OpenRouter expects ``vendor/model`` (e.g.
# ``z-ai/glm-5.2``), so we strip the LiteLLM router segment (``openrouter/``).
_LITELLM_PROVIDER_PREFIXES = ("openai/", "openrouter/", "azure/", "anthropic/", "google/")


def strip_litellm_provider_prefix(model: str) -> str:
    """Drop a leading LiteLLM provider segment for OpenAI-compatible proxies."""
    if not model or "/" not in model:
        return model
    lower = model.lower()
    for prefix in _LITELLM_PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            return model[len(prefix):]
    return model


def ensure_openai_provider_prefix(model: str) -> str:
    """FEDOT ``MASConfig`` requires a provider prefix; bare ids get ``openai/``.

    Keep ``openai/<id>`` / ``openrouter/<vendor>/<id>`` in validated configs;
    ``_install_proxy_model_strip`` removes the LiteLLM segment on the wire.
    """
    if not model or not isinstance(model, str) or "/" in model:
        return model
    return f"openai/{model}"


def _prefix_bare_models_in_obj(obj: Any) -> Any:
    """Recursively rewrite bare ``model`` string fields to ``openai/...``."""
    if isinstance(obj, dict):
        return {
            k: ensure_openai_provider_prefix(v) if k == "model" and isinstance(v, str)
            else _prefix_bare_models_in_obj(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_prefix_bare_models_in_obj(item) for item in obj]
    return obj


def _install_proxy_model_strip() -> None:
    """Monkeypatch ``_ProxyClient.acompletion`` once so proxy calls strip providers."""
    if getattr(_ProxyClient.acompletion, "_coscientist_strips_provider", False):
        return
    original = _ProxyClient.acompletion

    async def acompletion(self, model, messages, tools, **kwargs):  # noqa: ANN001
        stripped = strip_litellm_provider_prefix(model)
        if stripped != model:
            _log.debug("PatchedMAS: proxy model %r → %r", model, stripped)
        return await original(self, stripped, messages, tools, **kwargs)

    acompletion._coscientist_strips_provider = True  # type: ignore[attr-defined]
    _ProxyClient.acompletion = acompletion  # type: ignore[method-assign]


def _install_parse_llm_output_unwrap() -> None:
    from fedotmas.meta._helpers import parse_llm_output as _orig

    if getattr(_orig, "_coscientist_unwraps_completion_state", False):
        return

    def parse_llm_output(raw: Any, schema: Any) -> Any:  # noqa: ANN001
        unwrapped = _unwrap_completion_state(raw)
        # Routing meta sometimes emits bare ids (``glm-4.7``); MASConfig
        # validation requires a provider prefix. Prefix before pydantic parse.
        if isinstance(unwrapped, (dict, list)):
            unwrapped = _prefix_bare_models_in_obj(unwrapped)
        elif isinstance(unwrapped, str):
            try:
                parsed = json.loads(unwrapped)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                unwrapped = json.dumps(_prefix_bare_models_in_obj(parsed), ensure_ascii=False)
        return _orig(unwrapped, schema)

    parse_llm_output._coscientist_unwraps_completion_state = True  # type: ignore[attr-defined]
    import fedotmas.meta._helpers as _helpers

    _helpers.parse_llm_output = parse_llm_output  # type: ignore[method-assign]


def ensure_fedot_openai_proxy_compat() -> None:
    """Install wire-strip + MASConfig model-prefix fixes (idempotent).

    Safe to call from vanilla ``MAS`` paths (clean arm) as well as ``PatchedMAS``.
    """
    _install_proxy_model_strip()
    _install_parse_llm_output_unwrap()


# Configurable guard against injecting an unbounded task into every worker.
_MAX_TASK_CHARS = int(os.getenv("COSCIENTIST_FEDOT_TASK_MAX_CHARS", "6000"))
_IDENTIFIER_RE = re.compile(r"(?<!\w)[a-z][a-z0-9]+(?:_[a-z0-9]+)+(?!\w)")

_NO_GREET_SUFFIX = (
    " The user task is already provided below. Do NOT greet or ask clarifying"
    " questions. Immediately use your tools to complete your part and return"
    " concrete results (SMILES, scores, URLs) — never an empty acknowledgment."
)

_COORDINATOR_TRANSFER_SUFFIX = (
    " When calling transfer_to_agent, assume the worker may NOT see prior chat."
    " Restate the USER TASK and pass any intermediate artifacts (SMILES lists,"
    " docking scores, S3 URLs) explicitly in your message. Never greet the user."
)


class PatchedMAS(MAS):
    """MAS with small CoScientist patches around ADK AutoFlow routing.

    1. Drop tool names that are not real MCP servers (meta sometimes puts
       ``transfer_to_agent`` in ``tools`` → ``Unknown MCP server``).
    2. Single worker → run it directly (skip coordinator / transfer).
    3. Multi-worker → inject the full user task into every agent instruction
       so a ``transfer_to_agent`` with ``result: null`` does not lose the task.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Install proxy/model compat explicitly (idempotent) rather than as an
        # import-time side effect, so importing this module is inert.
        ensure_fedot_openai_proxy_compat()
        super().__init__(*args, **kwargs)
        self._pending_task: str | None = None

    async def generate_config(self, task: str) -> MASConfig:
        self._pending_task = task
        return await super().generate_config(task)

    async def build_and_run(
        self,
        config: MASConfig,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self._pending_task = user_query
        return await super().build_and_run(config, user_query, initial_state=initial_state, timeout=timeout)

    def build(self, config: MASConfig) -> BaseAgent:
        registry = self._mcp_registry
        worker_models = self._worker_map()

        # Collapse to one worker BEFORE sanitize: sanitize injects ## USER TASK
        # into every worker (and may attach the same servers to empty-tool
        # workers), which would otherwise make all workers look equally relevant.
        picked = self._pick_single_worker(config)
        if picked is not None:
            config = config.model_copy(update={"workers": [picked]})
        config = self._sanitize_config(config)

        if len(config.workers) == 1:
            _log.info(
                "PatchedMAS: single worker %r — skipping coordinator/transfer",
                config.workers[0].name,
            )
            return _build_routing_agent(config.workers[0], registry, worker_models)

        # Sanitize can equalize toolsets (empty-tool workers get all servers);
        # retry the collapse once before falling back to full transfer routing.
        picked = self._pick_single_worker(config)
        if picked is not None:
            return _build_routing_agent(picked, registry, worker_models)

        _log.info(
            "PatchedMAS: building routing system (%d workers) with task injection",
            len(config.workers),
        )
        return build_routing_system(config, mcp_registry=registry, worker_models=worker_models)

    def _pick_single_worker(self, config: MASConfig) -> MASAgentConfig | None:
        """Reduce many workers to one when transfer routing adds no capability.

        Tried in order: (1) an unambiguous runtime-identifier match from the
        task, (2) the sole registered MCP server, (3) an identical toolset across
        every worker. Returns None when workers genuinely differ, so the full
        coordinator/transfer routing system is built. No repository-owned
        function or server allowlist is used.
        """
        workers = config.workers
        if len(workers) <= 1:
            return workers[0] if workers else None

        # (1) Runtime-identifier narrowing — only on an unambiguous winner, so a
        # named function is never attached to the wrong MCP server on a tie.
        identifiers = set(_IDENTIFIER_RE.findall((self._pending_task or "").lower()))
        if identifiers:
            scored = sorted(
                (
                    (
                        sum(
                            1
                            for ident in identifiers
                            if re.search(
                                rf"(?<!\w){re.escape(ident)}(?!\w)",
                                f"{w.name} {w.description} {w.instruction} {' '.join(w.tools)}".lower(),
                            )
                        ),
                        w,
                    )
                    for w in workers
                ),
                key=lambda sw: sw[0],
                reverse=True,
            )
            if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                _log.info(
                    "PatchedMAS: narrowing %d workers → %r from runtime identifiers %s",
                    len(workers), scored[0][1].name, sorted(identifiers),
                )
                return scored[0][1]

        # (2) A single registered MCP server ⇒ prefer the worker already listing it.
        known = list(getattr(self, "_mcp_registry", None) or {})
        if len(known) == 1:
            server = known[0]
            pick = next((w for w in workers if server in (w.tools or [])), workers[0])
            _log.info(
                "PatchedMAS: single MCP server %r — collapsing %d workers → %r",
                server, len(workers), pick.name,
            )
            return pick

        # (3) Identical toolset across all workers ⇒ transfers add no capability.
        signatures = {tuple(sorted(w.tools or [])) for w in workers}
        if len(signatures) == 1:
            _log.info(
                "PatchedMAS: all workers share tools %s — collapsing to %r",
                sorted(next(iter(signatures))), workers[0].name,
            )
            return workers[0]
        return None

    def _sanitize_config(self, config: MASConfig) -> MASConfig:
        # Registry keys are MCP *server names* (e.g. GenerativeMoleculeModels).
        # Meta may emit MCP *function* names where registry *server* names are
        # required; those must not leave the worker with an empty toolset.
        known = sorted(self._mcp_registry or {})
        known_set = set(known)
        task_block = self._task_block()

        def clean(agent: MASAgentConfig, *, is_coordinator: bool) -> MASAgentConfig:
            original = list(agent.tools)
            tools = [t for t in original if t in known_set]
            bad = [t for t in original if t not in known_set]
            if bad:
                _log.warning(
                    "PatchedMAS: dropping unknown tools %s from agent %r (known=%s)",
                    bad, agent.name, known,
                )
            # Workers with no valid server tools cannot call MCP — attach every
            # server we were given (CoScientist already filters to task-relevant
            # servers before constructing PatchedMAS).
            if not tools and known and not is_coordinator:
                tools = list(known)
                _log.warning(
                    "PatchedMAS: agent %r had no registry tools (had %s) — attaching %s so MCP can run",
                    agent.name, original, tools,
                )
            instruction = agent.instruction.rstrip()
            if tools and "Do NOT greet" not in instruction:
                instruction += _NO_GREET_SUFFIX
            if is_coordinator and "Restate the USER TASK" not in instruction:
                instruction += _COORDINATOR_TRANSFER_SUFFIX
            if task_block and "## USER TASK" not in instruction:
                instruction += task_block
            return agent.model_copy(update={"tools": tools, "instruction": instruction})

        return config.model_copy(
            update={
                "coordinator": clean(config.coordinator, is_coordinator=True),
                "workers": [clean(w, is_coordinator=False) for w in config.workers],
            }
        )

    def _task_block(self) -> str:
        task = (self._pending_task or "").strip()
        if not task:
            return ""
        if len(task) > _MAX_TASK_CHARS:
            task = task[:_MAX_TASK_CHARS] + "\n…[truncated]"
        return f"\n\n## USER TASK (authoritative — execute this; do not greet)\n{task}\n"
