"""Finalize a run into the report folder deliverable.

Runs in the manager driver AFTER the aggregator agent has produced its narrative
markdown. Writes ``report.md``, renders LaTeX per :class:`ReportConfig`, and
writes ``MANIFEST.json`` by scanning what actually landed on disk.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from CoScientist.config.report import ReportConfig
from CoScientist.reporting.collect import report_dir_for
from CoScientist.reporting.latex import render_latex

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """What :meth:`CoScientistManager.run` returns.

    ``markdown`` is the assembled report text (for a chat bubble / stdout).
    ``report_dir`` is the on-disk folder deliverable. ``manifest`` lists its
    contents. ``report_dir``/``manifest`` are ``None`` if no folder was written.
    """

    markdown: str
    report_dir: Optional[Path] = None
    manifest: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:  # so legacy `print(result)` / str() still reads well
        return self.markdown


def finalize_report(
    session_id: str,
    final_markdown: str,
    report_config: ReportConfig,
    state: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """Write report.md + LaTeX + MANIFEST.json; return a :class:`RunResult`."""
    report_dir = report_dir_for(session_id, report_config.reports_root)
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.md").write_text(final_markdown or "", encoding="utf-8")

        references = _extract_references(state or {})
        latex_files = render_latex(
            final_markdown or "", report_dir, report_config.latex, references
        )
        manifest = _build_manifest(session_id, report_dir, report_config, latex_files)
        (report_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("report: wrote deliverable to %s (latex=%s)", report_dir, report_config.latex)
        return RunResult(markdown=final_markdown, report_dir=report_dir, manifest=manifest)
    except Exception as exc:  # never let report packaging sink a completed run
        logger.error("report: failed to finalize %s (%s)", report_dir, exc)
        return RunResult(markdown=final_markdown, report_dir=None, manifest=None)


def _build_manifest(
    session_id: str,
    report_dir: Path,
    report_config: ReportConfig,
    latex_files: List[Path],
) -> Dict[str, Any]:
    def listing(subdir: str) -> List[str]:
        d = report_dir / subdir
        if not d.exists():
            return []
        return sorted(str(p.relative_to(report_dir)) for p in d.rglob("*") if p.is_file())

    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": "report.md" if (report_dir / "report.md").exists() else None,
        "figures": listing("figures"),
        "tables": listing("tables"),
        "sections": listing("sections"),
        "latex": {
            "mode": report_config.latex,
            "files": sorted(str(p.relative_to(report_dir)) for p in latex_files),
        },
    }


def _extract_references(state: Dict[str, Any]) -> List[str]:
    """Best-effort structured references from session state.

    TODO(bibliography): paper-research currently stores results as free text
    (``search_results``), so there is no reliable citation metadata to build a
    real ``references.bib`` from. When paper-research is changed to retain raw
    paper/citation records in state (e.g. a ``references`` list of dicts), read
    them here. Until then this returns whatever plain-string references are
    already present and otherwise nothing.
    """
    refs = state.get("references")
    if isinstance(refs, list):
        out: List[str] = []
        for r in refs:
            if isinstance(r, str):
                out.append(r)
            elif isinstance(r, dict):
                out.append(r.get("citation") or r.get("title") or json.dumps(r))
        return out
    return []


__all__ = ["finalize_report", "RunResult"]
