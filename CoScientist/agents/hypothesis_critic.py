from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 1
VERIF_REQUIRED = 1
MAX_BATCH_SIZE = 8
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5"

_SYSTEM = (
    "You are a scientific hypothesis critic. "
    "Respond ONLY with a JSON array - no prose, no markdown fences."
    "Each element corresponds to one input hypothesis (same order)."
    'Schema: {"id":"<str>","scores":{"verifiability":<0|1|2>,"consistency":<0|1|2>,'
    '"specificity":<0|1|2>,"novelty":<0|1|2>},"feedback":"<str|null>","tool_request":"<str|null>"}\n'
    "Scores: 0=fail 1=partial 2=pass.\n"
    "verifiability: 0=cannot test with listed tools, 1=testable but plan incomplete, 2=fully testable.\n"
    "consistency: 0=contradicts RAG context, 1=uncertain, 2=consistent.\n"
    "specificity: 0=vague/unfalsifiable, 1=partially specific, 2=clear falsifiable claim.\n"
    "novelty: 0=already published per RAG, 1=incremental, 2=novel.\n"
    "feedback: one concise sentence per failing dimension, null if all scores>=2.\n"
    "tool_request: if verifiability==0 describe the missing tool in <=15 words, else null."
)

@dataclass
class HypothesisInput:
    id: str
    claim: str
    domain: str = ""
    variables: str = ""
    verification_plan: str = ""
    tools: list[str] = field(default_factory=list)
    strategy_type: str = ""

    def to_critic_dict(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "domain": self.domain,
            "variables": self.variables,
            "verification_plan": self.verification_plan,
            "tools": self.tools,
            "strategy_type": self.strategy_type,
        }

@dataclass
class HypothesisCriticResult:
    id: str
    scores: dict[str, int]
    passed: bool
    feedback: Optional[str]
    tools_available: bool
    tool_request: Optional[str]


class RAGClient:
    def query_batch(self, queries: list[str]) -> list[str]:
        return [""] * len(queries)


class HypothesisCriticAgent:
    def __init__(
        self,
        rag_client: RAGClient,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        rag_query_fn: Optional[Callable[[HypothesisInput], str]] = None,
    ):
        self._rag = rag_client
        raw_model = model or os.environ.get("LLM__MAIN_MODEL", MODEL)
        # Strip openrouter/ prefix for direct HTTP requests to OpenRouter
        if raw_model.startswith("openrouter/"):
            raw_model = raw_model[len("openrouter/"):]
        self._model = raw_model
        self._max_tokens = max_tokens
        self._rag_query = rag_query_fn or self._default_query
        self._api_key = os.environ.get(
            "OPENROUTER_API_KEY", os.environ.get("LLM__OPENAI_API_KEY", "")
        )
        self._api_url = os.environ.get("LLM__MAIN_URL", "")

    def critique_batch(self, hypotheses: list[HypothesisInput]) -> list[HypothesisCriticResult]:
        results: list[HypothesisCriticResult] = []
        for i in range(0, len(hypotheses), MAX_BATCH_SIZE):
            results.extend(self._process(hypotheses[i : i + MAX_BATCH_SIZE]))
        return results

    def critique_one(self, hypothesis: HypothesisInput) -> HypothesisCriticResult:
        return self.critique_batch([hypothesis])[0]

    def _process(self, chunk: list[HypothesisInput]) -> list[HypothesisCriticResult]:
        contexts = self._rag.query_batch([self._rag_query(h) for h in chunk])

        blocks = [
            f"<hypothesis>\n{json.dumps(h.to_critic_dict(), ensure_ascii=False)}\n"
            f"<rag_context>{ctx[:600]}</rag_context>\n</hypothesis>"
            for h, ctx in zip(chunk, contexts)
        ]

        target_url = self._api_url or OPENROUTER_URL
        if target_url and not target_url.endswith("/chat/completions"):
            target_url = target_url.rstrip("/") + "/chat/completions"
        response = requests.post(
            target_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": "\n".join(blocks)},
                ],
            },
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            logger.warning("HypothesisCriticAgent: LLM returned null content")
            return [self._fallback(h) for h in chunk]
        raw = content.strip()
        if not raw:
            logger.warning("HypothesisCriticAgent: LLM returned empty content")
            return [self._fallback(h) for h in chunk]
        return self._parse(raw, chunk)

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Extract JSON from LLM response — strips ``` fences and parses."""
        t = text.strip()
        # Remove ``` fences
        if t.startswith("```"):
            t = t[3:]
            if t.startswith("json"):
                t = t[4:]
            # Find closing ```
            fence = t.rfind("```")
            if fence != -1:
                t = t[:fence]
            t = t.strip()
        # Try direct parse
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return None

    def _parse(self, raw: str, chunk: list[HypothesisInput]) -> list[HypothesisCriticResult]:
        items = self._extract_json(raw)
        if items is None or not isinstance(items, list):
            logger.warning(
                "HypothesisCriticAgent: failed to parse response as JSON array. "
                "Raw (first 200 chars): %s", raw[:200]
            )
            return [self._fallback(h) for h in chunk]

        parsed = {item["id"]: item for item in items}
        results = []

        for h in chunk:
            item = parsed.get(h.id)
            if item is None:
                results.append(self._fallback(h))
                continue

            scores = {
                dim: max(0, min(2, int(item.get("scores", {}).get(dim, 0))))
                for dim in ("verifiability", "consistency", "specificity", "novelty")
            }
            passed = (
                all(v >= PASS_THRESHOLD for v in scores.values())
                and scores["verifiability"] >= VERIF_REQUIRED
            )
            results.append(HypothesisCriticResult(
                id=h.id,
                scores=scores,
                passed=passed,
                feedback=item.get("feedback"),
                tools_available=scores["verifiability"] >= 1,
                tool_request=item.get("tool_request"),
            ))

        return results

    @staticmethod
    def _default_query(h: HypothesisInput) -> str:
        return f"{h.claim} {h.domain}"[:300]

    @staticmethod
    def _fallback(h: HypothesisInput) -> HypothesisCriticResult:
        return HypothesisCriticResult(
            id=h.id,
            scores={"verifiability": 0, "consistency": 0, "specificity": 0, "novelty": 0},
            passed=False,
            feedback="Critic failed to evaluate this hypothesis.",
            tools_available=False,
            tool_request=None,
        )


def apply_critic_results(
    hypotheses: list[dict[str, Any]],
    results: list[HypothesisCriticResult],
    id_field: str = "id",
) -> list[dict[str, Any]]:
    result_map = {r.id: r for r in results}
    updated = []

    for hyp in hypotheses:
        hyp = dict(hyp)
        cr = result_map.get(hyp.get(id_field, ""))
        if cr is None:
            updated.append(hyp)
            continue

        hyp["critic"] = {
            "scores": cr.scores,
            "pass": cr.passed,
            "feedback": cr.feedback,
            "tools_available": cr.tools_available,
            "tool_request": cr.tool_request,
        }
        hyp["status"] = (
            "active" if cr.passed
            else "deferred" if not cr.tools_available
            else "proposed"
        )
        updated.append(hyp)

    return updated


