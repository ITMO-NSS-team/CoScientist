#!/usr/bin/env python3
"""Collect a small **error-statistics** table for the alembic benchmark run.

Two compact, side-by-side panels, both computed from a run's ``summary.json``:

  * **Failure classes** -- the taxonomy of runtime exceptions the pipeline hit
    while building / executing tools, from ``aggregate.failures_by_class``
    (cross-checked against the per-repo ``pipeline_metrics.failures_by_class``).
    Columns: class, count, share of all failures.
  * **Guard aborts** -- why a stage's agent loop was force-stopped by the cycle
    guard, from per-repo ``pipeline_metrics.abort_reason_per_stage``. Reasons are
    ``tool_cycle`` (same call repeated non-consecutively), ``tool_repeat`` (same
    call repeated consecutively) and ``max_steps`` (step budget exhausted),
    split by the stage they fired in.

Emits ``docs/paper/tables/alembic_error_stats.tex`` (standalone-compilable) and a
markdown preview. No paper references.

Usage
-----
    python collect_error_stats_table.py [RUN_DIR] [--out PATH.tex] [--no-markdown]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
DOCS_DIR = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]
DEFAULT_RUN = REPO_ROOT / "benchmarks/alembic/runs/2026-07-10_tmbench-all-v2"
DEFAULT_OUT = DOCS_DIR / "paper/tables/alembic_error_stats.tex"

PIPELINE_STAGES = ("explorer", "environment", "coder", "validator", "wrapper")
# human-readable one-liners for the guard-abort reasons (used in the caption)
REASON_GLOSS = {
    "tool_cycle": "same tool call repeated non-consecutively",
    "tool_repeat": "same tool call repeated consecutively",
    "max_steps": "step budget exhausted",
}


def collect(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    repos = summary.get("repos", [])
    agg = summary.get("aggregate", {})

    # ---- failure classes (prefer aggregate, cross-check per-repo) ----
    fbc = dict(agg.get("failures_by_class") or {})
    recomputed = Counter()
    for r in repos:
        for cls, n in ((r.get("pipeline_metrics") or {}).get("failures_by_class") or {}).items():
            recomputed[cls] += n
    if not fbc:
        fbc = dict(recomputed)
    elif dict(recomputed) and Counter(fbc) != recomputed:
        print(f"  ! failures_by_class mismatch: aggregate={dict(fbc)} "
              f"per-repo={dict(recomputed)} (using aggregate)", file=sys.stderr)

    # ---- guard aborts: reason -> stage -> count ----
    reason_stage = defaultdict(Counter)
    for r in repos:
        for stage, reason in ((r.get("pipeline_metrics") or {}).get("abort_reason_per_stage") or {}).items():
            if reason:
                reason_stage[reason][stage] += 1

    return summary, fbc, reason_stage


def sorted_failures(fbc):
    # count desc, then name asc for determinism
    return sorted(fbc.items(), key=lambda kv: (-kv[1], kv[0]))


def active_stages(reason_stage):
    stages = {s for cnts in reason_stage.values() for s in cnts}
    return [s for s in PIPELINE_STAGES if s in stages]


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
TEX_PREAMBLE = r"""\documentclass{article}
% Standalone preview of the alembic error-statistics table. Compile:
%   pdflatex alembic_error_stats
% The two `tabular`s can be \input into the main paper individually.
\usepackage[a4paper,margin=1.5cm]{geometry}
\usepackage{booktabs}

\begin{document}
\pagestyle{empty}
"""


def build_latex(summary, fbc, reason_stage):
    fails = sorted_failures(fbc)
    total_fail = sum(n for _, n in fails)
    stages = active_stages(reason_stage)
    reasons = sorted(reason_stage, key=lambda r: (-sum(reason_stage[r].values()), r))
    stage_tot = {s: sum(reason_stage[r].get(s, 0) for r in reasons) for s in stages}
    grand_abort = sum(stage_tot.values())

    L = [TEX_PREAMBLE, r"\begin{table}[t]", r"\centering", r"\footnotesize"]

    # ---- panel 1: failure classes ----
    L.append(r"\begin{minipage}[t]{0.42\linewidth}\centering")
    L.append(r"\textbf{(a) Failure classes}\\[2pt]")
    L.append(r"\begin{tabular}{@{}lr@{}}")
    L.append(r"\toprule")
    L.append(r"Error class & Count \\")
    L.append(r"\midrule")
    for cls, n in fails:
        L.append(rf"{cls} & {n} \\")
    L.append(r"\midrule")
    L.append(rf"\textbf{{Total}} & \textbf{{{total_fail}}} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{minipage}\hfill")

    # ---- panel 2: guard aborts ----
    col_spec = "@{}l" + "r" * len(stages) + "r@{}"
    L.append(r"\begin{minipage}[t]{0.50\linewidth}\centering")
    L.append(r"\textbf{(b) Guard aborts by stage}\\[2pt]")
    L.append(rf"\begin{{tabular}}{{{col_spec}}}")
    L.append(r"\toprule")
    header = ["Reason"] + [s.capitalize() for s in stages] + ["Total"]
    L.append(" & ".join(header) + r" \\")
    L.append(r"\midrule")
    for r in reasons:
        row_tot = sum(reason_stage[r].values())
        cells = [r.replace("_", r"\_")] + [str(reason_stage[r].get(s, 0)) for s in stages] + [str(row_tot)]
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule")
    tot_cells = [r"\textbf{Total}"] + [rf"\textbf{{{stage_tot[s]}}}" for s in stages] + [rf"\textbf{{{grand_abort}}}"]
    L.append(" & ".join(tot_cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{minipage}")

    n_repos = summary.get("aggregate", {}).get("repos_with_data", len(summary.get("repos", [])))
    L.append(
        rf"\caption{{Error statistics across the {n_repos} repositories. "
        rf"\textbf{{(a)}} runtime exceptions by class ({total_fail} total). "
        rf"\textbf{{(b)}} cycle-guard aborts ({grand_abort} total): the reason each "
        rf"stage's inner agent loop was force-stopped, recorded once per stage, "
        rf"distinct from the stage-gate retries, which count separately.}}"
    )
    L.append(r"\label{tab:alembic-error-stats}")
    L.append(r"\end{table}")
    L.append(r"\end{document}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def build_markdown(fbc, reason_stage):
    fails = sorted_failures(fbc)
    total_fail = sum(n for _, n in fails)
    stages = active_stages(reason_stage)
    reasons = sorted(reason_stage, key=lambda r: (-sum(reason_stage[r].values()), r))

    out = ["**(a) Failure classes**", "", "| Error class | Count |", "|---|---|"]
    for cls, n in fails:
        out.append(f"| {cls} | {n} |")
    out.append(f"| **Total** | **{total_fail}** |")

    out += ["", "**(b) Guard aborts by stage**", ""]
    head = "| Reason | " + " | ".join(s.capitalize() for s in stages) + " | Total |"
    out += [head, "|" + "|".join(["---"] * (len(stages) + 2)) + "|"]
    stage_tot = {s: 0 for s in stages}
    grand = 0
    for r in reasons:
        rt = sum(reason_stage[r].values())
        grand += rt
        for s in stages:
            stage_tot[s] += reason_stage[r].get(s, 0)
        out.append("| " + r + " | " + " | ".join(str(reason_stage[r].get(s, 0)) for s in stages)
                   + f" | {rt} |")
    out.append("| **Total** | " + " | ".join(f"**{stage_tot[s]}**" for s in stages)
               + f" | **{grand}** |")
    return "\n".join(out)


def paper_body(latex):
    """Extract an \\input-able ``table*`` fragment from the standalone LaTeX."""
    import re
    body = latex.split(r"\pagestyle{empty}", 1)[-1].split(r"\end{document}", 1)[0]
    body = re.sub(r"\\begin\{table\}(\[[^\]]*\])?", r"\\begin{table*}[t]", body)
    body = body.replace(r"\end{table}", r"\end{table*}")
    return ("% Auto-generated fragment — \\input this into the paper; do not edit by hand.\n"
            + body.strip() + "\n")


# short stage labels so panel (b) fits a single column
SHORT_STAGE = {"explorer": "Expl.", "environment": "Env.", "coder": "Coder",
               "validator": "Valid.", "wrapper": "Wrap."}
_AUTOGEN = "% Auto-generated fragment — \\input this into the paper; do not edit by hand.\n"


def paper_body_failures(summary, fbc):
    """Single-column ``table`` fragment: runtime exceptions by class."""
    fails = sorted_failures(fbc)
    total = sum(n for _, n in fails)
    n_repos = summary.get("aggregate", {}).get("repos_with_data", len(summary.get("repos", [])))
    L = [_AUTOGEN, r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Error class & Count \\", r"\midrule"]
    L += [rf"{cls} & {n} \\" for cls, n in fails]
    L += [r"\midrule", rf"\textbf{{Total}} & \textbf{{{total}}} \\", r"\bottomrule",
          r"\end{tabular}",
          rf"\caption{{Runtime exceptions by class across the {n_repos} "
          rf"repositories ({total} total).}}",
          r"\label{tab:alembic-failures}", r"\end{table}", ""]
    return "\n".join(L)


def paper_body_aborts(reason_stage):
    """Single-column ``table`` fragment: cycle-guard aborts by stage."""
    stages = active_stages(reason_stage)
    reasons = sorted(reason_stage, key=lambda r: (-sum(reason_stage[r].values()), r))
    stage_tot = {s: sum(reason_stage[r].get(s, 0) for r in reasons) for s in stages}
    grand = sum(stage_tot.values())
    col_spec = "@{}l" + "r" * len(stages) + "r@{}"
    header = ["Reason"] + [SHORT_STAGE.get(s, s.capitalize()) for s in stages] + ["Total"]
    L = [_AUTOGEN, r"\begin{table}[t]", r"\centering", r"\small",
         rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    for r in reasons:
        cells = [r.replace("_", r"\_")] + [str(reason_stage[r].get(s, 0)) for s in stages] \
            + [str(sum(reason_stage[r].values()))]
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(" & ".join([r"\textbf{Total}"] + [rf"\textbf{{{stage_tot[s]}}}" for s in stages]
                        + [rf"\textbf{{{grand}}}"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Cycle-guard aborts by stage ({grand} total): the reason each "
          rf"stage's inner agent loop was force-stopped, recorded once per stage, "
          rf"distinct from the stage-gate retries, which count separately.}}",
          r"\label{tab:alembic-aborts}", r"\end{table}", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=str(DEFAULT_RUN),
                    help=f"benchmark run dir with summary.json (default: {DEFAULT_RUN})")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-markdown", dest="markdown", action="store_false")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "summary.json").exists():
        ap.error(f"no summary.json in {run_dir}")

    summary, fbc, reason_stage = collect(run_dir)
    latex = build_latex(summary, fbc, reason_stage)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    out.with_name(out.stem + "_body.tex").write_text(paper_body(latex), encoding="utf-8")
    # split, single-column fragments (the paper \inputs these two separately)
    out.with_name(out.stem + "_a_body.tex").write_text(
        paper_body_failures(summary, fbc), encoding="utf-8")
    out.with_name(out.stem + "_b_body.tex").write_text(
        paper_body_aborts(reason_stage), encoding="utf-8")
    total_fail = sum(fbc.values())
    total_abort = sum(sum(c.values()) for c in reason_stage.values())
    print(f"[error-stats] run:   {run_dir}")
    print(f"[error-stats] LaTeX: {out}")
    print(f"[error-stats] {total_fail} failures across {len(fbc)} classes; "
          f"{total_abort} guard aborts")

    if args.markdown:
        print()
        print(build_markdown(fbc, reason_stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
