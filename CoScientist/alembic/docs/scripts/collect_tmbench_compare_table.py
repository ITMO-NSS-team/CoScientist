#!/usr/bin/env python3
"""Collect the TM-Bench-*validated* comparison table: alembic (ours) vs ToolMaker.

Counterpart to ``docs/paper/tables/toolmaker_initial.tex`` (which put ToolMaker on
the left and OpenHands on the right). Here:

  * left half  = **alembic (ours)**, scored on the *real* TM-Bench pytest suite;
  * right half = **ToolMaker**, the published baseline numbers (kept verbatim
    from ``toolmaker_initial.tex``);
  * rows are **per task** (15 TM-Bench tasks), no per-row paper references;
  * columns per side are **Invoc. | Tests | Tokens** -- Cost (price) and Actions
    are dropped, Tokens kept. Our Tokens is the *total pipeline token usage across
    all stages* for the repo that produced the task.

Our Invoc./Tests come from the JUnit XML that the TM-Bench harness produced for
our exported tools, in ``benchmarks/alembic/TMBench/alembic_results/<task>.xml``:

  * **Tests**  = testcases passed / total in the suite.
  * **Invoc.** = distinct invocations whose every testcase passed / total invocations
    (an invocation is a ``property name="invocation"`` group).

The denominators reproduce TM-Bench's official invocation/test counts, so the two
halves are directly comparable.

Our Tokens come from the canonical benchmark run's ``summary.json``
(``pipeline_metrics.total_tokens`` per repo). STAMP contributes two tasks from a
single pipeline run, so its token count is shared (daggered, counted once in the
total).

Usage
-----
    python collect_tmbench_compare_table.py [--results DIR] [--run RUN_DIR]
                                            [--out PATH.tex] [--no-markdown]
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
DOCS_DIR = SCRIPT.parents[1]                       # .../alembic/docs
REPO_ROOT = SCRIPT.parents[4]                      # outer CoScientist checkout
DEFAULT_RESULTS = REPO_ROOT / "benchmarks/alembic/TMBench/alembic_results"
DEFAULT_RUN = REPO_ROOT / "benchmarks/alembic/runs/2026-07-10_tmbench-all-v2"
DEFAULT_OUT = DOCS_DIR / "paper/tables/alembic_tmbench_compare.tex"

SUBSECTIONS = ("Pathology", "Rad.", "Omics", "Other")

# Per-task metadata + ToolMaker baseline numbers (verbatim from toolmaker_initial.tex).
# key = XML stem = tool/task name; repo = which pipeline run supplied our tokens.
#   tm = (invoc_passed, invoc_total, tests_passed, tests_total, tokens)
TASKS = [
    # Pathology
    dict(key="conch_extract_features",            sub="Pathology", repo="CONCH",
         tm=(3, 3, 9, 9, 171226)),
    dict(key="musk_extract_features",             sub="Pathology", repo="MUSK",
         tm=(3, 3, 6, 6, 696386)),
    dict(key="pathfinder_verify_biomarker",       sub="Pathology", repo="PathFinderCRC",
         tm=(0, 2, 4, 6, 356825)),
    dict(key="stamp_extract_features",            sub="Pathology", repo="STAMP",
         tm=(3, 3, 12, 12, 631138)),
    dict(key="stamp_train_classification_model",  sub="Pathology", repo="STAMP",
         tm=(3, 3, 9, 9, 1249521)),
    dict(key="uni_extract_features",              sub="Pathology", repo="UNI",
         tm=(3, 3, 9, 9, 326806)),
    # Radiology
    dict(key="medsam_inference",                  sub="Rad.", repo="MedSAM",
         tm=(3, 3, 6, 6, 508954)),
    dict(key="nnunet_train_model",                sub="Rad.", repo="nnUNet",
         tm=(0, 2, 0, 4, 1792291)),
    # Omics
    dict(key="cytopus_db",                        sub="Omics", repo="cytopus",
         tm=(3, 3, 12, 12, 185912)),
    dict(key="esm_fold_predict",                  sub="Omics", repo="esm",
         tm=(2, 3, 13, 15, 336754)),
    # Other
    dict(key="flowmap_overfit_scene",             sub="Other", repo="flowmap",
         tm=(2, 2, 6, 6, 358552)),
    dict(key="medsss_generate",                   sub="Other", repo="MedSSS",
         tm=(3, 3, 6, 6, 282771)),
    dict(key="modernbert_predict_masked",         sub="Other", repo="ModernBERT",
         tm=(3, 3, 9, 9, 356228)),
    dict(key="retfound_feature_vector",           sub="Other", repo="RETFound_MAE",
         tm=(3, 3, 6, 6, 561936)),
    dict(key="tabpfn_predict",                    sub="Other", repo="TabPFN",
         tm=(3, 3, 9, 9, 95257)),
]


# --------------------------------------------------------------------------- #
# TM-Bench JUnit XML parsing
# --------------------------------------------------------------------------- #
def parse_junit(xml_path: Path):
    """Return (invoc_passed, invoc_total, tests_passed, tests_total)."""
    root = ET.parse(xml_path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    tests_total = tests_passed = 0
    inv_ok = OrderedDict()          # invocation -> all-passed so far?
    for tc in suite.findall("testcase"):
        props_el = tc.find("properties")
        props = {p.get("name"): p.get("value")
                 for p in (props_el if props_el is not None else ())}
        inv = props.get("invocation", tc.get("name", "?"))
        failed = (tc.find("failure") is not None
                  or tc.find("error") is not None
                  or tc.find("skipped") is not None)
        tests_total += 1
        if not failed:
            tests_passed += 1
        inv_ok[inv] = inv_ok.get(inv, True) and not failed
    inv_total = len(inv_ok)
    inv_passed = sum(1 for ok in inv_ok.values() if ok)
    return inv_passed, inv_total, tests_passed, tests_total


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def frac_class(num, den):
    if den in (None, 0):
        return None
    if num >= den:
        return "G"
    if num <= 0:
        return "R"
    return "Y"


def tex_frac(num, den):
    body = f"{num}/{den}"
    c = frac_class(num, den)
    return f"\\{c}{{{body}}}" if c else body


def fmt_tokens(n):
    return "--" if n is None else f"{n:,}"


def md_frac(num, den):
    return f"{num}/{den}"


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
TEX_PREAMBLE = r"""\documentclass{article}
% Standalone preview of the alembic-vs-ToolMaker TM-Bench-validated table.
% Compile: pdflatex alembic_tmbench_compare. The `table` environment can be
% \input into the main paper as-is (drop the \documentclass ... \end{document}).
\usepackage[a4paper,margin=1.2cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}
\usepackage[table]{xcolor}
\usepackage{amssymb}
\usepackage{pifont}
\usepackage{graphicx}  % \rotatebox

% ---- cell colours (all pass / partial / none) ----
\definecolor{okgreen}{RGB}{198,239,206}
\definecolor{midyellow}{RGB}{255,235,156}
\definecolor{badred}{RGB}{255,199,206}
\newcommand{\G}[1]{\cellcolor{okgreen}#1}
\newcommand{\Y}[1]{\cellcolor{midyellow}#1}
\newcommand{\R}[1]{\cellcolor{badred}#1}

\begin{document}
\pagestyle{empty}
"""


def latex_task_name(key):
    return key.replace("_", r"\_")


def build_latex(rows, totals):
    L = [TEX_PREAMBLE,
         r"\begin{table}", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4.5pt}", r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{tabular}{@{}ll rrr rrr@{}}", r"\toprule",
         r"& & \multicolumn{3}{c}{\textbf{alembic (ours)}} "
         r"& \multicolumn{3}{c}{\textbf{ToolMaker} \small(W\"olflein et al., 2025)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"& Task & Invoc. & Tests & Tokens & Invoc. & Tests & Tokens \\",
         r"\midrule"]

    for si, sub in enumerate(SUBSECTIONS):
        block = [r for r in rows if r["sub"] == sub]
        if not block:
            continue
        n = len(block)
        for ri, r in enumerate(block):
            first = (rf"\multirow{{{n}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{sub}}}}}"
                     if ri == 0 else "")
            our_tok = fmt_tokens(r["our_tok"])
            if r["shared"]:
                our_tok += r"$^\dagger$"
            cells = [
                first,
                latex_task_name(r["key"]),
                tex_frac(r["our_iv"], r["our_it"]),
                tex_frac(r["our_tp"], r["our_tt"]),
                our_tok,
                tex_frac(r["tm_iv"], r["tm_it"]),
                tex_frac(r["tm_tp"], r["tm_tt"]),
                fmt_tokens(r["tm_tok"]),
            ]
            L.append(" & ".join(cells) + r" \\")
        if si != len(SUBSECTIONS) - 1:
            L.append(r"\midrule")

    t = totals
    L.append(r"\midrule")
    L.append(" & ".join([
        "",
        r"\textbf{Total}",
        rf"\textbf{{{t['our_iv']}/{t['our_it']}}}",
        rf"\textbf{{{t['our_tp']}/{t['our_tt']}}}",
        rf"\textbf{{{fmt_tokens(t['our_tok'])}}}",
        rf"\textbf{{{t['tm_iv']}/{t['tm_it']}}}",
        rf"\textbf{{{t['tm_tp']}/{t['tm_tt']}}}",
        rf"\textbf{{{fmt_tokens(t['tm_tok'])}}}",
    ]) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(
        r"\caption{TM-Bench-validated comparison against the published "
        r"\textsc{ToolMaker} baseline, per task, scored on the shared pytest suite; "
        r"green/yellow/red mark all/some/none passing. $^\dagger$STAMP's two tasks "
        r"share one Alembic run (tokens counted once).}"
    )
    L.append(r"\label{tab:alembic-tmbench-compare}")
    L.append(r"\end{table}")
    L.append(r"\end{document}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Markdown preview
# --------------------------------------------------------------------------- #
def build_markdown(rows, totals):
    out = ["| Subsec | Task | Ours Invoc | Ours Tests | Ours Tokens "
           "| TM Invoc | TM Tests | TM Tokens |",
           "|" + "|".join(["---"] * 8) + "|"]
    for r in rows:
        tok = fmt_tokens(r["our_tok"]) + ("†" if r["shared"] else "")
        out.append("| " + " | ".join([
            r["sub"], r["key"],
            md_frac(r["our_iv"], r["our_it"]), md_frac(r["our_tp"], r["our_tt"]), tok,
            md_frac(r["tm_iv"], r["tm_it"]), md_frac(r["tm_tp"], r["tm_tt"]),
            fmt_tokens(r["tm_tok"]),
        ]) + " |")
    t = totals
    out.append("| " + " | ".join([
        "**Total**", "",
        f"**{t['our_iv']}/{t['our_it']}**", f"**{t['our_tp']}/{t['our_tt']}**",
        f"**{fmt_tokens(t['our_tok'])}**",
        f"**{t['tm_iv']}/{t['tm_it']}**", f"**{t['tm_tp']}/{t['tm_tt']}**",
        f"**{fmt_tokens(t['tm_tok'])}**",
    ]) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def collect(results_dir: Path, run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    repo_tokens = {rec.get("repo"): (rec.get("pipeline_metrics") or {}).get("total_tokens")
                   for rec in summary.get("repos", [])}
    repo_task_count = {}
    for tk in TASKS:
        repo_task_count[tk["repo"]] = repo_task_count.get(tk["repo"], 0) + 1

    rows, missing = [], []
    for tk in TASKS:
        xml = results_dir / f"{tk['key']}.xml"
        if not xml.exists():
            missing.append(tk["key"])
            iv = it = tp = tt = 0
        else:
            iv, it, tp, tt = parse_junit(xml)
        tm_iv, tm_it, tm_tp, tm_tt, tm_tok = tk["tm"]
        rows.append(dict(
            key=tk["key"], sub=tk["sub"], repo=tk["repo"],
            our_iv=iv, our_it=it, our_tp=tp, our_tt=tt,
            our_tok=repo_tokens.get(tk["repo"]),
            shared=repo_task_count[tk["repo"]] > 1,
            tm_iv=tm_iv, tm_it=tm_it, tm_tp=tm_tp, tm_tt=tm_tt, tm_tok=tm_tok,
        ))
    if missing:
        print(f"  ! missing result XML for: {', '.join(missing)}", file=sys.stderr)

    totals = dict(our_iv=0, our_it=0, our_tp=0, our_tt=0, our_tok=0,
                  tm_iv=0, tm_it=0, tm_tp=0, tm_tt=0, tm_tok=0)
    counted_repos = set()
    for r in rows:
        for k in ("our_iv", "our_it", "our_tp", "our_tt",
                  "tm_iv", "tm_it", "tm_tp", "tm_tt", "tm_tok"):
            totals[k] += r[k] or 0
        if r["repo"] not in counted_repos:       # tokens counted once per repo run
            totals["our_tok"] += r["our_tok"] or 0
            counted_repos.add(r["repo"])
    return rows, totals


def paper_body(latex):
    """Extract an \\input-able single-column fragment (resized to \\columnwidth)."""
    import re
    body = latex.split(r"\pagestyle{empty}", 1)[-1].split(r"\end{document}", 1)[0]
    body = re.sub(r"\\begin\{table\}(\[[^\]]*\])?", r"\\begin{table}[t]", body)
    body = body.replace(r"\begin{tabular}", "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}")
    body = body.replace(r"\end{tabular}", "\\end{tabular}}")
    return ("% Auto-generated fragment — \\input this into the paper; do not edit by hand.\n"
            + body.strip() + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help=f"dir of TM-Bench JUnit XMLs (default: {DEFAULT_RESULTS})")
    ap.add_argument("--run", default=str(DEFAULT_RUN),
                    help=f"benchmark run dir for our tokens (default: {DEFAULT_RUN})")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-markdown", dest="markdown", action="store_false")
    args = ap.parse_args(argv)

    results_dir = Path(args.results).resolve()
    run_dir = Path(args.run).resolve()
    if not results_dir.is_dir():
        ap.error(f"results dir not found: {results_dir}")
    if not (run_dir / "summary.json").exists():
        ap.error(f"no summary.json in {run_dir}")

    rows, totals = collect(results_dir, run_dir)

    latex = build_latex(rows, totals)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    out.with_name(out.stem + "_body.tex").write_text(paper_body(latex), encoding="utf-8")
    print(f"[tmbench-compare] results: {results_dir}")
    print(f"[tmbench-compare] run:     {run_dir}")
    print(f"[tmbench-compare] LaTeX:   {out}  ({len(rows)} tasks)")
    print(f"[tmbench-compare] ours  Invoc {totals['our_iv']}/{totals['our_it']}  "
          f"Tests {totals['our_tp']}/{totals['our_tt']}")
    print(f"[tmbench-compare] TM    Invoc {totals['tm_iv']}/{totals['tm_it']}  "
          f"Tests {totals['tm_tp']}/{totals['tm_tt']}")

    if args.markdown:
        print()
        print(build_markdown(rows, totals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
