"""Shared Experiment Module runtime constants and soft-match helpers."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

MOLECULE_GENERATOR_TOOLS = frozenset(
    {"generate_case_mols", "generate_mols", "generate_molecules"}
)

# Admissions that a "success" was built on fake/proxy evidence → force partial.
FABRICATION_MARKERS = re.compile(
    r"(?i)\b("
    r"simulated|simulation|hardcoded|hard-coded|fabricated|fabrication|"
    r"placeholder|mock(?:ed)?|synthetic\s+dataset|fake\s+citation|"
    r"invented\s+(?:findings?|citations?|data)|as\s+if\s+(?:the\s+)?image|"
    r"no\s+real\s+(?:pubmed|search|api)|literature\s+data\s+was\s+simulated|"
    r"simulate_p(?:ubmed)?"
    r")\b"
)

_NAME_KEY = re.compile(r"[^a-z0-9]+")


def artifact_name_key(name: str) -> str:
    """Normalize artifact basename stem for soft matching (csv ≈ candidates.csv)."""
    return _NAME_KEY.sub("", Path(str(name)).stem.lower())


def audit(
    log: logging.Logger,
    message: str,
    *,
    stdout: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit an Experiment Module audit marker.

    Writes ``message`` to the caller's logger (preserving per-module logger
    names) and mirrors it to stdout when COSCIENTIST_EXPERIMENT_AUDIT_STDOUT=1.
    ``stdout`` overrides the console line for markers whose smoke-script format
    intentionally differs from the log line (e.g. adds ``ids=[...]``).
    """
    log.log(level, message)
    if os.getenv("COSCIENTIST_EXPERIMENT_AUDIT_STDOUT") == "1":
        print(message if stdout is None else stdout, flush=True)


__all__ = [
    "FABRICATION_MARKERS",
    "MOLECULE_GENERATOR_TOOLS",
    "artifact_name_key",
    "audit",
]
