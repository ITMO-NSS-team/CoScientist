# ToolMaker benchmark results

Snapshot of the most recent bench run over the ToolMaker subset — 14 repos
spanning pathology / medical imaging / NLP / classical ML / structural
& cell biology.

To (re)populate this file, run:

```
python benchmarks/alembic/run_benchmark.py \
    --repos-file benchmarks/alembic/toolmaker_subset.txt \
    --parallel 4 \
    --output benchmarks/alembic/toolmaker_results.md
```

Once the bench completes, the summary table and per-repo details are written
here, overwriting the placeholder. Subset composition is defined in
`subset.py` (`TOOLMAKER_REPOS`) and resolved to URLs via
`repository_inventory.jsonl`.

<!-- Below this line — bench summary will be written by run_benchmark.py -->
