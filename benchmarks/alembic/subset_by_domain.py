"""Sample N repos per domain randomly (fixed seed) from repository_inventory.jsonl."""

import json
import random
from collections import defaultdict
from pathlib import Path

N    = 2
SEED = 42
base = Path(__file__).parent
input_path  = base / "repository_inventory.jsonl"
output_path = base / "toolrosella_subset.txt"

rows = [json.loads(line) for line in input_path.read_text().splitlines()]

by_domain: dict[str, list] = defaultdict(list)
for row in rows:
    by_domain[row["domain"]].append(row)

rng = random.Random(SEED)
subset = []
for domain, candidates in sorted(by_domain.items()):
    picked = rng.sample(candidates, min(N, len(candidates)))
    subset.extend(picked)

output_path.write_text("\n".join(r["github_url"] for r in subset) + "\n")
print(f"Saved {len(subset)} URLs ({N} per domain, seed={SEED}) to {output_path.name}")
for domain, candidates in sorted(by_domain.items()):
    print(f"  {min(N, len(candidates))}  {domain}")
