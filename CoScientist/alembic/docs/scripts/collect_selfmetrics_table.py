#!/usr/bin/env python3
"""Collect alembic's *self-metrics* benchmark table.

This is the counterpart to ``docs/paper/tables/toolmaker_initial.tex`` (the
ToolMaker-vs-OpenHands results table), but reports what the alembic pipeline
itself achieved on TM-Bench, with **one row per repository** (not per task).

Data source
-----------
A benchmark run directory produced by ``benchmarks/alembic/run_benchmark.py``,
i.e. a folder containing ``summary.json`` with ``{"repos": [...], "aggregate": {...}}``.
Each repo record carries ``stage_status`` (per-stage pass/fail + gates),
``validation.counts`` (tools/tests/exec/invoc tallies), ``pipeline_metrics``
(``total_tokens`` etc.) and ``elapsed_sec``.

Columns (mirrors the benchmark reports, with the tweaks requested)
-----------------------------------------------------------------
Reports had:  Repo | Time | Exit | Stage reached | Tools p/pf/t | Tests | Exec | Invoc
Here we emit: Repo | Stable | Pass-Invoc | Total | Tests | Exec | Invoc | All-stages? | Time | Tokens

  * ``Tools p/pf/t`` is split into three columns:
      - Stable      = tools_passed   ("tool imports/runs stably and its unit tests pass")
      - Pass-Invoc  = tools_perfect  ("also passes its live invocation example")
      - Total       = tools_total
  * ``Exit`` + ``Stage reached`` collapse into a single **All stages passed?** tick:
      - a green check when the pipeline delivered a working MCP end-to-end
        (coder + validator + wrapper passed and exit 0);
      - a *written* note (not a tick) for the single case whose environment stage
        only resolved after a non-fatal **soft** dependency conflict (TabPFN, whose
        sklearn private-API imports warn but do not block);
      - a red cross when no working MCP was produced.
    A spurious *hard* env-gate flag that was recovered downstream (e.g.
    PathFinderCRC's malformed ``import code/...`` smoke test) does not by itself
    count as a failure -- delivery is judged on coder+validator+wrapper+exit.
  * **Time** and **Tokens** (total pipeline token usage) are added.

Rows are grouped into the same subsections as the reference table
(Pathology / Rad. / Omics / Other).

Papers
------
Every row cites its source paper. ``--ensure-bib`` (default on) idempotently
appends any missing BibTeX entries -- copied from ToolMaker's ``benchmark/papers.bib``
and augmented with the paper URL from ``benchmark/papers.yaml`` -- to
``docs/paper/custom.bib``.

Usage
-----
    python collect_selfmetrics_table.py [RUN_DIR] [--out PATH.tex]
                                        [--no-ensure-bib] [--no-markdown]

With no RUN_DIR it defaults to the canonical run
``benchmarks/alembic/runs/2026-07-10_tmbench-all-v2``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Locations (resolved relative to this file so the script is checkout-portable)
# --------------------------------------------------------------------------- #
SCRIPT = Path(__file__).resolve()
DOCS_DIR = SCRIPT.parents[1]                       # .../alembic/docs
REPO_ROOT = SCRIPT.parents[4]                      # outer CoScientist checkout
DEFAULT_RUN = REPO_ROOT / "benchmarks/alembic/runs/2026-07-10_tmbench-all-v2"
DEFAULT_OUT = DOCS_DIR / "paper/tables/alembic_selfmetrics.tex"
BIB_PATH = DOCS_DIR / "paper/custom.bib"

PIPELINE_STAGES = ("explorer", "environment", "coder", "validator", "wrapper")

# --------------------------------------------------------------------------- #
# Repo -> (display label, subsection, [bib keys]).  Order defines table order.
# Subsections and grouping follow docs/paper/tables/toolmaker_initial.tex.
# --------------------------------------------------------------------------- #
SUBSECTIONS = ("Pathology", "Rad.", "Omics", "Other")

REPO_META = {
    # Pathology
    "CONCH":         ("CONCH",       "Pathology", ["lu2024conch"]),
    "MUSK":          ("MUSK",        "Pathology", ["xiang2025musk"]),
    "PathFinderCRC": ("PathFinder",  "Pathology", ["liang2023pathfinder"]),
    "STAMP":         ("STAMP",       "Pathology", ["elnahhas2024stamp"]),
    "UNI":           ("UNI",         "Pathology", ["chen2024uni"]),
    # Radiology
    "MedSAM":        ("MedSAM",      "Rad.",      ["ma2024medsam"]),
    "nnUNet":        ("nnU-Net",     "Rad.",      ["isensee2020nnunet"]),
    # Omics
    "cytopus":       ("Cytopus",     "Omics",     ["kunes2023cytopus"]),
    "esm":           ("ESM",         "Omics",     ["verkuil2022esm1", "hie2022esm2"]),
    # Other
    "flowmap":       ("FlowMap",     "Other",     ["smith2024flowmap"]),
    "MedSSS":        ("MedSSS",      "Other",     ["jiang2025medsss"]),
    "ModernBERT":    ("ModernBERT",  "Other",     ["warner2024modernbert"]),
    "RETFound_MAE":  ("RETFound",    "Other",     ["zhou2023retfound"]),
    "TabPFN":        ("TabPFN",      "Other",     ["hollmann2025tabpfn"]),
}

# --------------------------------------------------------------------------- #
# BibTeX entries (verbatim from ToolMaker benchmark/papers.bib) + url from
# benchmark/papers.yaml.  Only entries whose key is missing get appended.
# --------------------------------------------------------------------------- #
PAPERS = {
    "lu2024conch": r"""@article{lu2024conch,
  author    = {Lu, Ming Y. and Chen, Bowen and Williamson, Drew F. K. and Chen, Richard J. and Liang, Ivy and Ding, Tong and Jaume, Guillaume and Odintsov, Igor and Le, Long Phi and Gerber, Georg and Parwani, Anil V. and Zhang, Andrew and Mahmood, Faisal},
  title     = {A visual-language foundation model for computational pathology},
  year      = {2024},
  journal   = {Nature Medicine},
  volume    = {30},
  number    = {3},
  pages     = {863--874},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41591-024-02856-4},
}""",
    "xiang2025musk": r"""@article{xiang2025musk,
  author    = {Xiang, Jinxi and Wang, Xiyue and Zhang, Xiaoming and Xi, Yinghua and Eweje, Feyisope and Chen, Yijiang and Li, Yuchen and Bergstrom, Colin and Gopaulchan, Matthew and Kim, Ted and Yu, Kun-Hsing and Willens, Sierra and Olguin, Francesca Maria and Nirschl, Jeffrey J. and Neal, Joel and Diehn, Maximilian and Yang, Sen and Li, Ruijiang},
  title     = {A vision-language foundation model for precision oncology},
  year      = {2025},
  journal   = {Nature},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41586-024-08378-w},
}""",
    "liang2023pathfinder": r"""@article{liang2023pathfinder,
  author    = {Liang, Junhao and Zhang, Weisheng and Yang, Jianghui and Wu, Meilong and Dai, Qionghai and Yin, Hongfang and Xiao, Ying and Kong, Lingjie},
  title     = {Deep learning supported discovery of biomarkers for clinical prognosis of liver cancer},
  year      = {2023},
  journal   = {Nature Machine Intelligence},
  volume    = {5},
  number    = {4},
  pages     = {408--420},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s42256-023-00635-3},
}""",
    "elnahhas2024stamp": r"""@article{elnahhas2024stamp,
  author    = {El Nahhas, Omar S. M. and van Treeck, Marko and W\"{o}lflein, Georg and Unger, Michaela and Ligero, Marta and Lenz, Tim and Wagner, Sophia J. and Hewitt, Katherine J. and Khader, Firas and Foersch, Sebastian and Truhn, Daniel and Kather, Jakob Nikolas},
  title     = {From whole-slide image to biomarker prediction: end-to-end weakly supervised deep learning in computational pathology},
  year      = {2024},
  journal   = {Nature Protocols},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41596-024-01047-2},
}""",
    "chen2024uni": r"""@article{chen2024uni,
  author    = {Chen, Richard J. and Ding, Tong and Lu, Ming Y. and Williamson, Drew F. K. and Jaume, Guillaume and Song, Andrew H. and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Shaban, Muhammad and Williams, Mane and Oldenburg, Lukas and Weishaupt, Luca L. and Wang, Judy J. and Vaidya, Anurag and Le, Long Phi and Gerber, Georg and Sahai, Sharifa and Williams, Walt and Mahmood, Faisal},
  title     = {Towards a general-purpose foundation model for computational pathology},
  year      = {2024},
  journal   = {Nature Medicine},
  volume    = {30},
  number    = {3},
  pages     = {850--862},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41591-024-02857-3},
}""",
    "ma2024medsam": r"""@article{ma2024medsam,
  author    = {Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  title     = {Segment anything in medical images},
  year      = {2024},
  journal   = {Nature Communications},
  volume    = {15},
  number    = {1},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41467-024-44824-z},
}""",
    "isensee2020nnunet": r"""@article{isensee2020nnunet,
  author    = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  title     = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  year      = {2020},
  journal   = {Nature Methods},
  volume    = {18},
  number    = {2},
  pages     = {203--211},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41592-020-01008-z},
}""",
    "kunes2023cytopus": r"""@article{kunes2023cytopus,
  author    = {Kunes, Russell Z. and Walle, Thomas and Land, Max and Nawy, Tal and Pe'er, Dana},
  title     = {Supervised discovery of interpretable gene programs from single-cell data},
  year      = {2023},
  journal   = {Nature Biotechnology},
  volume    = {42},
  number    = {7},
  pages     = {1084--1095},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41587-023-01940-3},
}""",
    "verkuil2022esm1": r"""@misc{verkuil2022esm1,
  author        = {Verkuil, Robert and Kabeli, Ori and Du, Yilun and Wicky, Basile I. M. and Milles, Lukas F. and Dauparas, Justas and Baker, David and Ovchinnikov, Sergey and Sercu, Tom and Rives, Alexander},
  title         = {Language models generalize beyond natural proteins},
  year          = {2022},
  archiveprefix = {bioRxiv},
  eprint        = {2022.12.21.521521},
  url           = {https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1},
}""",
    "hie2022esm2": r"""@misc{hie2022esm2,
  author        = {Hie, Brian and Candido, Salvatore and Lin, Zeming and Kabeli, Ori and Rao, Roshan and Smetanin, Nikita and Sercu, Tom and Rives, Alexander},
  title         = {A high-level programming language for generative protein design},
  year          = {2022},
  archiveprefix = {bioRxiv},
  eprint        = {2022.12.21.521526},
  url           = {https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1},
}""",
    "smith2024flowmap": r"""@misc{smith2024flowmap,
  author        = {Cameron Smith and David Charatan and Ayush Tewari and Vincent Sitzmann},
  title         = {FlowMap: High-Quality Camera Poses, Intrinsics, and Depth via Gradient Descent},
  year          = {2024},
  archiveprefix = {arXiv},
  eprint        = {2404.15259},
  url           = {https://arxiv.org/abs/2404.15259},
}""",
    "jiang2025medsss": r"""@misc{jiang2025medsss,
  author        = {Shuyang Jiang and Yusheng Liao and Zhe Chen and Ya Zhang and Yanfeng Wang and Yu Wang},
  title         = {MedS$^3$: Towards Medical Small Language Models with Self-Evolved Slow Thinking},
  year          = {2025},
  archiveprefix = {arXiv},
  eprint        = {2501.12051},
  url           = {https://arxiv.org/abs/2501.12051},
}""",
    "warner2024modernbert": r"""@misc{warner2024modernbert,
  author        = {Benjamin Warner and Antoine Chaffin and Benjamin Clavi\'{e} and Orion Weller and Oskar Hallstr\"{o}m and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
  title         = {Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
  year          = {2024},
  archiveprefix = {arXiv},
  eprint        = {2412.13663},
  url           = {https://arxiv.org/abs/2412.13663},
}""",
    "zhou2023retfound": r"""@article{zhou2023retfound,
  author    = {Zhou, Yukun and Chia, Mark A. and Wagner, Siegfried K. and Ayhan, Murat S. and Williamson, Dominic J. and Struyven, Robbert R. and Liu, Timing and Xu, Moucheng and Lozano, Mateo G. and Woodward-Court, Peter and others},
  title     = {A foundation model for generalizable disease detection from retinal images},
  year      = {2023},
  journal   = {Nature},
  volume    = {622},
  number    = {7981},
  pages     = {156--163},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41586-023-06555-x},
}""",
    "hollmann2025tabpfn": r"""@article{hollmann2025tabpfn,
  author    = {Hollmann, Noah and M\"{u}ller, Samuel and Purucker, Lennart and Krishnakumar, Arjun and K\"{o}rfer, Max and Hoo, Shi Bin and Schirrmeister, Robin Tibor and Hutter, Frank},
  title     = {Accurate predictions on small data with a tabular foundation model},
  year      = {2025},
  journal   = {Nature},
  volume    = {637},
  number    = {8045},
  pages     = {319--326},
  publisher = {Springer Science and Business Media LLC},
  url       = {https://www.nature.com/articles/s41586-024-08328-6},
}""",
}


# --------------------------------------------------------------------------- #
# Metric extraction
# --------------------------------------------------------------------------- #
def _st(rec, stage):
    return ((rec.get("stage_status") or {}).get(stage) or {}).get("status")


def stage_outcome(rec):
    """('pass'|'soft'|'fail', detail) for the *All stages passed?* column."""
    delivered = (
        rec.get("exit_code") == 0
        and _st(rec, "coder") == "passed"
        and _st(rec, "validator") == "passed"
        and _st(rec, "wrapper") == "passed"
    )
    if not delivered:
        # what broke first, for the detail note
        broke = next((s for s in PIPELINE_STAGES if _st(rec, s) != "passed"), "?")
        return "fail", f"{broke} stage failed"
    env_soft = (((rec.get("stage_status") or {}).get("environment") or {}).get("gate") or {}).get("soft") or []
    if env_soft:
        return "soft", f"{len(env_soft)} soft env conflict(s)"
    return "pass", ""


def counts(rec):
    val = rec.get("validation") or {}
    c = dict(val.get("counts") or {})
    # A tool marked *failed* must not be credited with passing invocations: a
    # tool whose exec check errored (e.g. ModernBERT's masked-prediction tool,
    # which rejected a malformed smoke input) is a failure, so its invocation
    # passes are counted as 0 rather than shown green beside the failed verdict.
    # invoc_total is left unchanged (the invocations were still attempted).
    demote = sum((t.get("invoc_passed") or 0) for t in (val.get("tools") or [])
                 if t.get("passed") is False)
    if demote and c.get("invoc_passed") is not None:
        c["invoc_passed"] = max(0, c["invoc_passed"] - demote)
    return c


def total_tokens(rec):
    return (rec.get("pipeline_metrics") or {}).get("total_tokens")


def stage_retries(rec):
    """Total guard-triggered stage resets (retries) across all stages."""
    ss = rec.get("stage_status") or {}
    return sum((ss.get(s) or {}).get("resets") or 0 for s in PIPELINE_STAGES)


# --------------------------------------------------------------------------- #
# Formatting helpers (shared by LaTeX + markdown)
# --------------------------------------------------------------------------- #
def fmt_time(sec):
    if sec is None:
        return "--"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def fmt_tokens(n):
    return "--" if n is None else f"{n:,}"


def frac_class(num, den):
    """'G' all / 'Y' some / 'R' none / None neutral (nothing attempted)."""
    if num is None or den in (None, 0):
        return None
    if num >= den:
        return "G"
    if num <= 0:
        return "R"
    return "Y"


# --------------------------------------------------------------------------- #
# LaTeX rendering
# --------------------------------------------------------------------------- #
TEX_PREAMBLE = r"""\documentclass{article}
% Standalone preview of the alembic self-metrics table. Compile with
%   pdflatex alembic_selfmetrics && bibtex alembic_selfmetrics && pdflatex alembic_selfmetrics x2
% to resolve the \citep links. The `table` environment can also be \input into
% the main paper (which already loads natbib + custom.bib), in which case drop
% the surrounding \documentclass ... \end{document} wrapper.
\usepackage[a4paper,margin=1.2cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}  % \makecell line breaks in header cells
\usepackage[table]{xcolor}
\usepackage{amssymb}
\usepackage{pifont}    % \ding{51} check, \ding{55} cross
\usepackage{graphicx}  % \rotatebox
\usepackage[numbers]{natbib}
\usepackage{hyperref}

% ---- cell colours (all pass / partial / none) ----
\definecolor{okgreen}{RGB}{198,239,206}
\definecolor{midyellow}{RGB}{255,235,156}
\definecolor{badred}{RGB}{255,199,206}
\newcommand{\G}[1]{\cellcolor{okgreen}#1}
\newcommand{\Y}[1]{\cellcolor{midyellow}#1}
\newcommand{\R}[1]{\cellcolor{badred}#1}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}

\begin{document}
\pagestyle{empty}
"""


def tex_color(body, cls):
    return f"\\{cls}{{{body}}}" if cls else body


def tex_int_cell(num, den):
    """Integer coloured by num/den ratio (used for the two Tools columns)."""
    if num is None:
        return "--"
    return tex_color(str(num), frac_class(num, den))


def tex_perfect_cell(num, den, invoc_total):
    """Pass-Invoc (tools_perfect). When the repo had *no* invocation tests at all
    (invoc_total == 0), 'perfect' cannot be assessed, so render the 0 neutral
    (uncoloured) rather than red -- it is N/A, not a failure."""
    if num is None:
        return "--"
    if not invoc_total:
        return str(num)
    return tex_color(str(num), frac_class(num, den))


def tex_frac_cell(num, den):
    if num is None or den is None:
        return "--"
    if den == 0:
        return "--"          # nothing attempted -> neutral
    return tex_color(f"{num}/{den}", frac_class(num, den))


def tex_allpass_cell(outcome):
    kind, _detail = outcome
    if kind == "pass":
        return r"\G{\cmark}"
    if kind == "soft":
        return r"\Y{soft-conf.}"   # the "written" (non-tick) case: TabPFN
    return r"\R{\xmark}"


def build_latex(records_by_sub, totals):
    lines = [TEX_PREAMBLE]
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    # subsection | repo | stable pass-invoc total | tests exec invoc | allpass | time retries tokens
    lines.append(r"\begin{tabular}{@{}ll rrr rrr c rrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{3}{c}{\textbf{Tools}} "
        r"& \multicolumn{3}{c}{\textbf{Checks}} & & & & \\"
    )
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    lines.append(
        r"& Repo & Stable & \makecell{Pass\\Invoc.} & Total "
        r"& Tests & Exec & Invoc & \makecell{All\\stages?} & Time & \makecell{Stage\\retries} & Tokens \\"
    )
    lines.append(r"\midrule")

    for si, sub in enumerate(SUBSECTIONS):
        rows = records_by_sub.get(sub) or []
        if not rows:
            continue
        n = len(rows)
        for ri, r in enumerate(rows):
            c = counts(r["rec"])
            ttot = c.get("tools_total")
            first = (
                rf"\multirow{{{n}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{sub}}}}}"
                if ri == 0 else ""
            )
            cite = ",".join(r["cite"])
            repo_cell = rf"{r['label']} \small\citep{{{cite}}}"
            cells = [
                first,
                repo_cell,
                tex_int_cell(c.get("tools_passed"), ttot),
                tex_perfect_cell(c.get("tools_perfect"), ttot, c.get("invoc_total")),
                str(ttot) if ttot is not None else "--",
                tex_frac_cell(c.get("tests_passed"), c.get("tests_total")),
                tex_frac_cell(c.get("exec_ok"), c.get("exec_attempted")),
                tex_frac_cell(c.get("invoc_passed"), c.get("invoc_total")),
                tex_allpass_cell(r["outcome"]),
                fmt_time(r["rec"].get("elapsed_sec")),
                str(stage_retries(r["rec"])),
                fmt_tokens(total_tokens(r["rec"])),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if si != len(SUBSECTIONS) - 1:
            lines.append(r"\midrule")

    # ---- totals row ----
    lines.append(r"\midrule")
    t = totals
    if t["soft"] or t["failed"]:
        allpass = rf"\textbf{{{t['clean']}\,\cmark}}"
        if t["soft"]:
            allpass += rf"\,+\,{t['soft']}\,soft"
        if t["failed"]:
            allpass += rf"\,+\,{t['failed']}\,\xmark"
    else:
        allpass = rf"\textbf{{{t['clean']}/{t['n']}}}"
    tot_cells = [
        "",
        rf"\textbf{{Total}} ({t['n']} repos)",
        rf"\textbf{{{t['tools_passed']}}}",
        rf"\textbf{{{t['tools_perfect']}}}",
        rf"\textbf{{{t['tools_total']}}}",
        rf"\textbf{{{t['tests_passed']}/{t['tests_total']}}}",
        rf"\textbf{{{t['exec_ok']}/{t['exec_attempted']}}}",
        rf"\textbf{{{t['invoc_passed']}/{t['invoc_total']}}}",
        allpass,
        rf"\textbf{{{fmt_time(t['time'])}}}",
        rf"\textbf{{{t['retries']}}}",
        rf"\textbf{{{fmt_tokens(t['tokens'])}}}",
    ]
    lines.append(" & ".join(tot_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Alembic self-metrics on the TM-Bench repositories, one row per "
        r"repository; green/yellow/red mark all/some/none passing. \emph{All stages?} "
        r"marks an end-to-end delivery (\cmark); TabPFN's environment resolved "
        r"only after a soft dependency conflict (\emph{soft-conf.}).}"
    )
    lines.append(r"\label{tab:alembic-selfmetrics}")
    lines.append(r"\end{table}")
    lines.append(r"")
    lines.append(r"\bibliographystyle{plainnat}")
    lines.append(r"\bibliography{../custom}")
    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Markdown rendering (quick console/report preview)
# --------------------------------------------------------------------------- #
def md_allpass(outcome):
    kind, detail = outcome
    if kind == "pass":
        return "✅"
    if kind == "soft":
        return f"⚠️ soft ({detail})"
    return f"❌ ({detail})"


def build_markdown(records_by_sub, totals):
    hdr = ("| Subsec | Repo | Stable | PassInvoc | Total | Tests | Exec | Invoc "
           "| All stages? | Time | Stage retries | Tokens |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    out = [hdr, sep]

    def frac(num, den):
        if num is None or den is None:
            return "--"
        if den == 0:
            return "--"
        return f"{num}/{den}"

    for sub in SUBSECTIONS:
        for r in records_by_sub.get(sub) or []:
            c = counts(r["rec"])
            out.append("| " + " | ".join([
                sub,
                r["label"],
                str(c.get("tools_passed", "--")),
                str(c.get("tools_perfect", "--")),
                str(c.get("tools_total", "--")),
                frac(c.get("tests_passed"), c.get("tests_total")),
                frac(c.get("exec_ok"), c.get("exec_attempted")),
                frac(c.get("invoc_passed"), c.get("invoc_total")),
                md_allpass(r["outcome"]),
                fmt_time(r["rec"].get("elapsed_sec")),
                str(stage_retries(r["rec"])),
                fmt_tokens(total_tokens(r["rec"])),
            ]) + " |")

    t = totals
    allpass = f"{t['clean']} ✅"
    if t["soft"]:
        allpass += f" + {t['soft']} ⚠️"
    if t["failed"]:
        allpass += f" + {t['failed']} ❌"
    out.append("| " + " | ".join([
        "**Total**", f"**{t['n']} repos**",
        f"**{t['tools_passed']}**", f"**{t['tools_perfect']}**", f"**{t['tools_total']}**",
        f"**{t['tests_passed']}/{t['tests_total']}**",
        f"**{t['exec_ok']}/{t['exec_attempted']}**",
        f"**{t['invoc_passed']}/{t['invoc_total']}**",
        f"**{allpass}**",
        f"**{fmt_time(t['time'])}**", f"**{t['retries']}**", f"**{fmt_tokens(t['tokens'])}**",
    ]) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# BibTeX maintenance
# --------------------------------------------------------------------------- #
def ensure_bib(bib_path: Path, keys) -> list[str]:
    """Append any missing PAPERS[key] entries to custom.bib. Returns keys added."""
    text = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
    added = []
    chunks = []
    for key in keys:
        if key not in PAPERS:
            print(f"  ! no BibTeX on file for '{key}' -- skipping", file=sys.stderr)
            continue
        if f"{{{key}," in text or f"{{{key} " in text:
            continue
        chunks.append(PAPERS[key])
        added.append(key)
    if chunks:
        block = ("\n\n% ---- TM-Bench source papers (alembic self-metrics table) ----\n"
                 + "\n\n".join(chunks) + "\n")
        with bib_path.open("a", encoding="utf-8") as fh:
            fh.write(block)
    return added


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def collect(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    by_repo = {r.get("repo"): r for r in summary.get("repos", [])}

    records_by_sub = {s: [] for s in SUBSECTIONS}
    used_keys = []
    totals = dict(n=0, tools_passed=0, tools_perfect=0, tools_total=0,
                  tests_passed=0, tests_total=0, exec_ok=0, exec_attempted=0,
                  invoc_passed=0, invoc_total=0,
                  delivered=0, clean=0, soft=0, failed=0, time=0, retries=0, tokens=0)

    missing = []
    for repo, (label, sub, cite) in REPO_META.items():
        rec = by_repo.get(repo)
        if rec is None:
            missing.append(repo)
            continue
        outcome = stage_outcome(rec)
        records_by_sub[sub].append(
            {"repo": repo, "label": label, "cite": cite, "rec": rec, "outcome": outcome}
        )
        used_keys.extend(cite)

        c = counts(rec)
        totals["n"] += 1
        for k in ("tools_passed", "tools_perfect", "tools_total",
                  "tests_passed", "tests_total", "exec_ok", "exec_attempted",
                  "invoc_passed", "invoc_total"):
            totals[k] += c.get(k) or 0
        if outcome[0] == "pass":
            totals["clean"] += 1
            totals["delivered"] += 1
        elif outcome[0] == "soft":
            totals["soft"] += 1
            totals["delivered"] += 1
        else:
            totals["failed"] += 1
        totals["time"] += rec.get("elapsed_sec") or 0
        totals["retries"] += stage_retries(rec)
        totals["tokens"] += total_tokens(rec) or 0

    if missing:
        print(f"  ! run dir is missing repos: {', '.join(missing)}", file=sys.stderr)
    return records_by_sub, totals, used_keys


def paper_body(latex, *, resize=False, drop_bib=False):
    """Extract an \\input-able ``table*`` fragment from the standalone LaTeX,
    so the EMNLP paper and the standalone preview share one source of truth."""
    import re
    body = latex.split(r"\pagestyle{empty}", 1)[-1].split(r"\end{document}", 1)[0]
    if drop_bib:
        body = (body.replace(r"\bibliographystyle{plainnat}", "")
                    .replace(r"\bibliography{../custom}", ""))
    body = re.sub(r"\\begin\{table\}(\[[^\]]*\])?", r"\\begin{table*}[t]", body)
    body = body.replace(r"\end{table}", r"\end{table*}")
    if resize:  # very wide tables → scale the tabular to the text block
        body = body.replace(r"\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}")
        body = body.replace(r"\end{tabular}", "\\end{tabular}}")
    return ("% Auto-generated fragment — \\input this into the paper; do not edit by hand.\n"
            + body.strip() + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=str(DEFAULT_RUN),
                    help=f"benchmark run dir with summary.json (default: {DEFAULT_RUN})")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"LaTeX output path (default: {DEFAULT_OUT})")
    ap.add_argument("--no-ensure-bib", dest="ensure_bib", action="store_false",
                    help="do not append missing paper entries to custom.bib")
    ap.add_argument("--no-markdown", dest="markdown", action="store_false",
                    help="do not print the markdown preview to stdout")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "summary.json").exists():
        ap.error(f"no summary.json in {run_dir}")

    records_by_sub, totals, used_keys = collect(run_dir)

    latex = build_latex(records_by_sub, totals)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    body_path = out.with_name(out.stem + "_body.tex")
    body_path.write_text(paper_body(latex, resize=True, drop_bib=True), encoding="utf-8")
    print(f"[selfmetrics] run:   {run_dir}")
    print(f"[selfmetrics] LaTeX: {out}  ({totals['n']} repos, "
          f"{totals['delivered']} delivered)")
    print(f"[selfmetrics] body:  {body_path}  (\\input-able table* fragment)")

    if args.ensure_bib:
        added = ensure_bib(BIB_PATH, dict.fromkeys(used_keys))  # de-dup, keep order
        if added:
            print(f"[selfmetrics] bib:   +{len(added)} entries -> {BIB_PATH.name}: "
                  f"{', '.join(added)}")
        else:
            print(f"[selfmetrics] bib:   all {len(set(used_keys))} entries already present")

    if args.markdown:
        print()
        print(build_markdown(records_by_sub, totals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
