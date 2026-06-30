"""Take the first N repos per domain from repository_inventory.jsonl."""

import json
from collections import defaultdict
from pathlib import Path

N = 2
base = Path(__file__).parent
input_path = base / "repository_inventory.jsonl"
output_path = base / "toolrosella_subset.txt"

rows = [json.loads(line) for line in input_path.read_text().splitlines()]

seen = defaultdict(int)
subset = []
for row in rows:
    domain = row["domain"]
    if seen[domain] < N:
        subset.append(row)
        seen[domain] += 1

output_path.write_text("\n".join(r["github_url"] for r in subset) + "\n")
print(f"Saved {len(subset)} URLs ({N} per domain) to {output_path.name}")
for domain, count in seen.items():
    print(f"  {count}  {domain}")
