#!/usr/bin/env bash
# Render all alembic paper tables to PDF (and a combined all_tables.pdf).
#
# Compiles each docs/paper/tables/*.tex in a throwaway temp dir (so no .aux/.log
# clutter lands in the repo) and copies the finished PDFs to
#   docs/paper/tables/build/
# The self-metrics table also gets the bibtex pass so its \citep links resolve.
#
#   docs/scripts/render_tables.sh            # render + build build/all_tables.pdf
#   docs/scripts/render_tables.sh --open     # also open the combined PDF
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TABLES="$(cd "$HERE/../paper/tables" && pwd)"
BUILD="$TABLES/build"
BIB="$TABLES/../custom.bib"

# tex stem -> needs bibtex? (1 = yes)
TEXES=(alembic_selfmetrics alembic_tmbench_compare alembic_error_stats)
NEEDS_BIB=(1 0 0)

command -v pdflatex >/dev/null || { echo "pdflatex not found (install a TeX distribution, e.g. texlive)"; exit 1; }

mkdir -p "$BUILD"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$BIB" "$TMP/custom.bib"

pdfs=()
for i in "${!TEXES[@]}"; do
    stem="${TEXES[$i]}"
    src="$TABLES/$stem.tex"
    [ -f "$src" ] || { echo "  ! missing $src (run the collect_*.py script first) — skipping"; continue; }
    echo "[render] $stem.tex"
    cp "$src" "$TMP/$stem.tex"
    ( cd "$TMP"
      pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >"$stem.log" 2>&1
      if [ "${NEEDS_BIB[$i]}" = "1" ]; then
          bibtex "$stem" >>"$stem.log" 2>&1 || true
          pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >>"$stem.log" 2>&1
          pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >>"$stem.log" 2>&1
      fi
    )
    if [ -f "$TMP/$stem.pdf" ]; then
        cp "$TMP/$stem.pdf" "$BUILD/$stem.pdf"
        echo "         -> build/$stem.pdf"
        pdfs+=("$BUILD/$stem.pdf")
    else
        echo "  ! $stem failed to compile — see $TMP kept? no; tail of log:"; tail -5 "$TMP/$stem.log" || true
    fi
done

if command -v pdfunite >/dev/null 2>&1 && [ "${#pdfs[@]}" -gt 1 ]; then
    pdfunite "${pdfs[@]}" "$BUILD/all_tables.pdf"
    echo "[render] combined -> build/all_tables.pdf"
fi

echo "[render] done. PDFs in: $BUILD"
if [ "${1:-}" = "--open" ]; then
    target="$BUILD/all_tables.pdf"; [ -f "$target" ] || target="${pdfs[0]:-}"
    [ -n "$target" ] && { command -v xdg-open >/dev/null && xdg-open "$target" >/dev/null 2>&1 || evince "$target" >/dev/null 2>&1 & }
fi
