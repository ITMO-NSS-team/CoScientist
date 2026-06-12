"""Strict-JSON plan generation with validate-then-repair (F015a / R05).

Generates an ``ExperimentPlan`` from a task by asking the LLM for JSON, validating
it against the Pydantic schema, and on failure feeding the error back for a bounded
number of repair rounds. This is the concrete "constrained decoding" mechanism the
plan thesis presupposes — flaky open models emit empty/invalid JSON (F014.A2), so
the repair loop is mandatory.

Provider pinning (F014): gpt-oss-120b is flaky on OpenRouter's default routing, so
we pin it to a known-good provider set unless told otherwise.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import litellm
from pydantic import ValidationError

from CoScientist.config import get_settings
from CoScientist.experiments.plan import ExperimentPlan, PlanError
from CoScientist.experiments.prompts import ToolInventory, build_planner_messages

settings = get_settings()
litellm.suppress_debug_info = True

# Known-good OpenRouter providers for gpt-oss-120b (see F014 / settings comment).
GPT_OSS_PROVIDERS = ["deepinfra", "groq", "together", "fireworks"]

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


class PlanGenerationError(RuntimeError):
    def __init__(self, task: str, errors: list[str], last_raw: str):
        self.errors = errors
        self.last_raw = last_raw
        super().__init__(
            f"failed to generate a valid plan after {len(errors)} attempt(s): {errors[-1] if errors else '?'}"
        )


def _extra_body(model: str, providers: Optional[list[str]]) -> Optional[dict]:
    provs = providers
    if provs is None and "gpt-oss" in (model or ""):
        provs = GPT_OSS_PROVIDERS
    if not provs:
        return None
    return {"provider": {"only": list(provs),
                         "allow_fallbacks": settings.llm.provider_allow_fallbacks}}


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text).strip()


def generate_plan(
    task: str,
    *,
    hypothesis: Optional[str] = None,
    literature: Optional[str] = None,
    tools: Optional[ToolInventory] = None,
    model: Optional[str] = None,
    providers: Optional[list[str]] = None,
    max_repairs: int = 2,
    temperature: float = 0.2,
) -> tuple[ExperimentPlan, dict]:
    """Generate a validated ExperimentPlan. Returns (plan, meta).

    meta = {attempts, model, raw_len, errors}. Raises PlanGenerationError if no
    valid plan is produced within max_repairs+1 attempts.
    """
    model = model or settings.llm.main_model
    api_key = settings.llm.openai_api_key
    extra = _extra_body(model, providers)
    messages = build_planner_messages(task, hypothesis=hypothesis, literature=literature, tools=tools)

    errors: list[str] = []
    raw = ""
    for attempt in range(max_repairs + 1):
        resp = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            num_retries=4,
            timeout=120,
            response_format={"type": "json_object"},
            extra_body=extra,
        )
        raw = _strip_fences((resp.choices[0].message.content or "").strip())
        try:
            plan = ExperimentPlan.model_validate_json(raw)
            return plan, {"attempts": attempt + 1, "model": model, "raw_len": len(raw), "errors": errors}
        except (ValidationError, PlanError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:400]}")
            messages = messages + [
                {"role": "assistant", "content": raw or "(empty response)"},
                {"role": "user",
                 "content": f"Your previous output was INVALID:\n{str(exc)[:600]}\n"
                            "Return ONLY a corrected JSON object matching the schema. No prose, no fences."},
            ]

    raise PlanGenerationError(task, errors, raw)
