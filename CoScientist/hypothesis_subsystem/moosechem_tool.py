"""
MooseChemTool — native Python hypothesis generation pipeline.

Replaces the subprocess/bash-based MOOSE-Chem runner with a pure Python
pipeline using litellm for LLM calls and HTTP for PubMed. No external
MOOSE-CHEM installation required.

Pipeline:
    1. Build literature corpus from research question (PubMed E-utilities)
    2. Generate hypotheses via LLM with corpus context
    3. Evaluate and score hypotheses via LLM
    4. Return structured ToolResult with Hypothesis Pydantic models
"""

from __future__ import annotations

import asyncio
import json
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import litellm

from CoScientist.config import get_settings
from CoScientist.hypothesis_subsystem.base_tool import BaseHypothesisTool
from CoScientist.hypothesis_subsystem.models import (
    AlternativeHypothesis,
    Hypothesis,
    HypothesisQuery,
    HypothesisStatus,
    Provenance,
    Reference,
    ScaleType,
    ToolResult,
    Variable,
    Variables,
)


_settings = get_settings()

# ---------------------------------------------------------------------------
# PubMed corpus builder (async, extracted from hypothesis-main/moosechem_tools)
# ---------------------------------------------------------------------------

class _AsyncCorpusBuilder:
    """Async refactor of the CorpusBuilder from hypothesis-main."""

    PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        model: str,
        max_papers_per_query: int = 10,
        base_url: Optional[str] = None,
    ):
        self._model = model
        self._max_papers = max_papers_per_query
        self._base_url = base_url or "https://api.openai.com/v1"

    # -- Query generation via LLM -------------------------------------------

    async def _generate_queries(
        self, research_question: str, background_survey: Optional[str]
    ) -> List[str]:
        """Generate PubMed search queries via LLM."""
        prompt = f"""You are a scientific literature search expert.

Given a research question and background survey, generate 12 PubMed search
queries to build a relevant inspiration corpus.

Rules:
- Queries must cover ADJACENT methodological areas, NOT direct answers.
- Do NOT repeat specific methods or terms already in the question/background.
- Queries should be broad and diverse.
- Each query must be 3-6 words.
- Return ONLY a JSON array of strings, nothing else.

Research question: {research_question}
Background survey: {background_survey or 'N/A'}

Example output format:
["query one here", "query two here", "query three here"]"""

        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                api_base=self._base_url,
            )
            content = resp["choices"][0]["message"]["content"].strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1 or end == 0:
                return [research_question]
            return json.loads(content[start:end])
        except Exception as exc:
            print(f"[MooseChemTool] Query generation via LLM failed: {exc}")
            print("[MooseChemTool] Falling back to research question as search term.")
            # Extract key terms from research question for better PubMed search
            words = research_question.split()
            fallback = research_question[:200]  # Truncate if very long
            return [fallback]

    # -- PubMed E-utilities -------------------------------------------------

    async def _search_pubmed(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """Search PubMed for paper IDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self._max_papers,
            "retmode": "json",
            "sort": "relevance",
        }
        try:
            async with session.get(
                self.PUBMED_SEARCH_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                return data.get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            print(f"  [PubMed search] Error for '{query}': {exc}")
            return []

    async def _fetch_abstracts(
        self, session: aiohttp.ClientSession, pmids: List[str]
    ) -> List[List[str]]:
        """Fetch title + abstract for a list of PubMed IDs."""
        if not pmids:
            return []
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }
        try:
            async with session.get(
                self.PUBMED_FETCH_URL, params=params, timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                xml_content = await resp.text()
        except Exception as exc:
            print(f"  [PubMed fetch] Error: {exc}")
            return []

        root = ET.fromstring(xml_content)
        papers: List[List[str]] = []
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                "".join(el.itertext()).strip() for el in abstract_parts
            ).strip()
            if title and abstract:
                papers.append([title, abstract])
        return papers

    # -- Build corpus -------------------------------------------------------

    async def build(
        self, research_question: str, background_survey: Optional[str] = None
    ) -> List[List[str]]:
        """Full async corpus building pipeline."""
        # Generate queries
        queries = await self._generate_queries(research_question, background_survey)

        # Fetch papers from PubMed
        all_papers: List[List[str]] = []
        seen_titles: Set[str] = set()

        async with aiohttp.ClientSession() as session:
            for query in queries:
                pmids = await self._search_pubmed(session, query)
                papers = await self._fetch_abstracts(session, pmids)
                for paper in papers:
                    if paper[0] not in seen_titles:
                        seen_titles.add(paper[0])
                        all_papers.append(paper)
                await asyncio.sleep(0.35)  # Rate limiting for NCBI

        return all_papers


# ---------------------------------------------------------------------------
# MooseChemTool
# ---------------------------------------------------------------------------

_HYPOTHESIS_GENERATION_SYSTEM = """You are an expert scientific hypothesis generator. 
Your task is to produce rigorous, falsifiable, well-structured scientific hypotheses.

For each hypothesis, you MUST output ALL of the following fields in the JSON:

- claim: Assertion like "Compounds with feature X demonstrate effect Y in domain Z"
- variables: Object with independent[], dependent[], covariates[] arrays.
  Each variable has: name, description, unit (or null), scale (nominal|ordinal|interval|ratio)
- domain: Applicability scope (compound class, conditions)
- reasoning: Full logical derivation from literature/data, including limitations
- evidence_basis: Array of {doi, url, title, description} from the provided corpus
- verification_plan: Concrete steps: data, models, metrics, protocol
- tools: Required tools (list of strings)
- refutation_conditions: Popperian criteria (e.g. "MAE > 0.5" or "R² < 0.3")
- competing_with: Array of {claim, distinguishing_observation}

Be rigorous. Every claim must be falsifiable."""

_HYPOTHESIS_SCORING_SYSTEM = """You are a scientific hypothesis evaluator.
Score each hypothesis on a 0-100 scale across these dimensions:

- novelty (0-25): How original is the claim?
- plausibility (0-25): Does the evidence/literature support it?
- falsifiability (0-25): Are refutation conditions measurable and concrete?
- utility (0-25): If confirmed, how impactful would this be?

Return a JSON array of {index: int, scores: {novelty, plausibility, falsifiability, utility}, total: int}."""


class MooseChemTool(BaseHypothesisTool):
    """
    Native Python hypothesis generation using the MOOSE-Chem methodology.

    Pipeline: corpus building → LLM generation → LLM scoring → Hypothesis models.
    No external MOOSE-CHEM installation or bash subprocess required.
    """

    strategy_type = "MooseChem"

    def __init__(
        self,
        model: Optional[str] = None,
        max_papers_per_query: int = 10,
        max_hypotheses: int = 5,
        temperature: float = 0.7,
    ):
        self._model = model or _settings.llm.main_model
        self._corpus_builder = _AsyncCorpusBuilder(
            model=self._model, max_papers_per_query=max_papers_per_query
        )
        self._max_hypotheses = max_hypotheses
        self._temperature = temperature

    # ------------------------------------------------------------------
    # BaseHypothesisTool contract
    # ------------------------------------------------------------------

    def validate_query(self, query: HypothesisQuery) -> bool:
        """Query is valid if it has a non-empty research question."""
        return bool(query.research_question.strip())

    async def invoke(self, query: HypothesisQuery) -> ToolResult:
        """
        Execute the full MooseChem pipeline.

        Args:
            query: Structured input with research question, background, constraints.

        Returns:
            ToolResult with generated Hypothesis objects.
        """
        start_time = time.monotonic()
        try:
            # 1. Build corpus
            corpus = await self._corpus_builder.build(
                query.research_question, query.background_survey
            )
            if not corpus:
                return ToolResult(
                    strategy_type=self.strategy_type,
                    hypotheses=[],
                    metadata={"corpus_size": 0, "warning": "No papers found"},
                    success=False,
                    error_message="PubMed corpus is empty; cannot generate hypotheses.",
                )

            # 2. Generate hypotheses
            max_h = query.max_hypotheses if query.max_hypotheses else self._max_hypotheses
            raw_hypotheses = await self._generate_hypotheses(query, corpus, max_h)

            # 3. Score hypotheses
            scored = await self._score_hypotheses(raw_hypotheses)

            # 4. Build Pydantic models (sorted by score)
            scored.sort(key=lambda x: x.get("total", 0), reverse=True)
            hypotheses = [
                self._to_hypothesis(
                    h, scored[i] if i < len(scored) else None, query
                )
                for i, h in enumerate(raw_hypotheses[:max_h])
            ]

            duration_ms = (time.monotonic() - start_time) * 1000
            return ToolResult(
                strategy_type=self.strategy_type,
                hypotheses=hypotheses,
                metadata={
                    "corpus_size": len(corpus),
                    "model": self._model,
                    "duration_ms": round(duration_ms, 1),
                    "temperature": query.temperature or self._temperature,
                },
                success=True,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            return ToolResult(
                strategy_type=self.strategy_type,
                hypotheses=[],
                metadata={"duration_ms": round(duration_ms, 1)},
                success=False,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal: generation
    # ------------------------------------------------------------------

    def _build_corpus_context(self, corpus: List[List[str]], max_tokens: int = 12000) -> str:
        """Build a compact corpus string for the LLM prompt."""
        entries: List[str] = []
        total_chars = 0
        for i, (title, abstract) in enumerate(corpus):
            entry = f"[{i + 1}] {title}\n{abstract}"
            total_chars += len(entry)
            if total_chars > max_tokens * 3:  # Rough char estimate
                break
            entries.append(entry)
        return "\n\n".join(entries)

    async def _generate_hypotheses(
        self,
        query: HypothesisQuery,
        corpus: List[List[str]],
        max_hypotheses: int,
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses via LLM with corpus context."""
        corpus_text = self._build_corpus_context(corpus)
        temp = query.temperature if query.temperature is not None else self._temperature

        user_prompt = f"""Research question: {query.research_question}

Domain constraints: {query.domain_constraints or 'None specified'}

Background: {query.background_survey or 'None provided'}

Literature corpus ({len(corpus)} papers):
{corpus_text}

Generate {max_hypotheses} distinct, falsifiable scientific hypotheses based on the corpus.
Return ONLY a JSON array where each element has the exact structure specified.
Do NOT wrap in markdown, just raw JSON."""

        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _HYPOTHESIS_GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4000,
                temperature=temp,
                response_format={"type": "json_object"},
            )
            content = resp["choices"][0]["message"]["content"].strip()
            parsed = json.loads(content)

            # Handle both {"hypotheses": [...]} and direct [...] formats
            if isinstance(parsed, dict):
                hypotheses = parsed.get("hypotheses", [])
            elif isinstance(parsed, list):
                hypotheses = parsed
            else:
                hypotheses = []
            return hypotheses[:max_hypotheses]

        except Exception as exc:
            print(f"[MooseChemTool] Hypothesis generation failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Internal: scoring
    # ------------------------------------------------------------------

    async def _score_hypotheses(
        self, hypotheses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score hypotheses via LLM."""
        if not hypotheses:
            return []

        # Build compact representation for scoring
        items = [
            {"index": i, "claim": h.get("claim", "")[:300]}
            for i, h in enumerate(hypotheses)
        ]
        user_prompt = json.dumps({"hypotheses": items})

        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _HYPOTHESIS_SCORING_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp["choices"][0]["message"]["content"].strip()
            parsed = json.loads(content)

            if isinstance(parsed, dict):
                return parsed.get("scores", [])
            elif isinstance(parsed, list):
                return parsed
            return []

        except Exception as exc:
            print(f"[MooseChemTool] Scoring failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Internal: conversion to Pydantic models
    # ------------------------------------------------------------------

    def _to_hypothesis(
        self,
        raw: Dict[str, Any],
        score: Optional[Dict[str, Any]],
        query: HypothesisQuery,
    ) -> Hypothesis:
        """Convert a raw dict from LLM into a Hypothesis Pydantic model."""

        # Variables
        vars_raw = raw.get("variables", {})
        variables = Variables(
            independent=[
                Variable(
                    name=v.get("name", "?"),
                    description=v.get("description", ""),
                    unit=v.get("unit"),
                    scale=ScaleType(v.get("scale", "nominal")),
                )
                for v in vars_raw.get("independent", [])
            ],
            dependent=[
                Variable(
                    name=v.get("name", "?"),
                    description=v.get("description", ""),
                    unit=v.get("unit"),
                    scale=ScaleType(v.get("scale", "nominal")),
                )
                for v in vars_raw.get("dependent", [])
            ],
            covariates=[
                Variable(
                    name=v.get("name", "?"),
                    description=v.get("description", ""),
                    unit=v.get("unit"),
                    scale=ScaleType(v.get("scale", "nominal")),
                )
                for v in vars_raw.get("covariates", [])
            ],
        )

        # Evidence basis
        evidence = [
            Reference(
                doi=r.get("doi"),
                url=r.get("url"),
                title=r.get("title", "Untitled reference"),
                description=r.get("description"),
            )
            for r in raw.get("evidence_basis", [])
        ]

        # Competing hypotheses
        competing = [
            AlternativeHypothesis(
                claim=c.get("claim", ""),
                distinguishing_observation=c.get("distinguishing_observation", ""),
            )
            for c in raw.get("competing_with", [])
        ]

        # Build reasoning with score if available
        reasoning = raw.get("reasoning", "")
        if score:
            reasoning += (
                f"\n\n[MooseChem Score: {score.get('total', 'N/A')}/100 "
                f"(novelty={score.get('scores', {}).get('novelty', '?')}, "
                f"plausibility={score.get('scores', {}).get('plausibility', '?')}, "
                f"falsifiability={score.get('scores', {}).get('falsifiability', '?')}, "
                f"utility={score.get('scores', {}).get('utility', '?')})]"
            )

        return Hypothesis(
            claim=raw.get("claim", "Untitled hypothesis"),
            variables=variables,
            domain=raw.get("domain", query.domain_constraints or "Not specified"),
            reasoning=reasoning,
            strategy_type=self.strategy_type,
            evidence_basis=evidence,
            verification_plan=raw.get("verification_plan", "Not specified"),
            tools=raw.get("tools", []),
            refutation_conditions=raw.get("refutation_conditions", "Not specified"),
            competing_with=competing,
            status=HypothesisStatus.PROPOSED,
            provenance=Provenance(creator=f"MooseChemTool/{self._model}"),
        )
