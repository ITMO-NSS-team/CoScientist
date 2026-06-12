#!/usr/bin/env python
"""R05 benchmark — generate structured plans for a few dataset_S tasks (F015a).

Exercises CoScientist.experiments.generate_plan end-to-end (real LLM, strict JSON,
validate-then-repair) on representative drug-design tasks from dataset_S.xlsx, then
checks each plan is a valid DAG and that each step binds its tools to an MCP server.
Reports capability gaps (server/tool the planner named that isn't in the inventory —
the future F015c "detect" + F015d/F015e "build" branch).

Usage: python scripts/experiments/r05_plan_benchmark.py [model]
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
from CoScientist.experiments import PlanGenerationError, generate_plan  # noqa: E402

# Representative SERVER-GROUPED inventory — a stand-in for the live tool index that
# retrieve_tools / F015c (roadmap R07) will provide. {server: [(tool, desc), ...]}.
INVENTORY = {
    "generative-models-mcp": [
        ("generate_case_mols", "generate candidate molecules for a known target case (GAN/Transformer)"),
        ("generate_mols", "sample molecules from a trained generative model"),
    ],
    "chemical-mcp-server": [
        ("calculate_docking", "docking score of a SMILES against a protein (pdb id)"),
        ("get_rdkit_properties", "RDKit physchem properties from SMILES (MW, logP, TPSA, QED)"),
        ("retrosynthesis_route", "retrosynthetic route search for a SMILES"),
    ],
    "admet-mcp": [
        ("predict_bbb_permeability", "predict blood-brain-barrier permeability from SMILES"),
        ("predict_admet", "predict ADMET / toxicity / solubility endpoints from SMILES"),
    ],
    "bioactivity-mcp": [
        ("fetch_protein_activities", "fetch known actives for a protein from ChEMBL/BindingDB"),
    ],
}

# Verbatim tasks from CoScientist/dataset_S.xlsx.
TASKS = [
    ("alzheimer/GSK-3beta",
     "Generate GSK-3beta inhibitors with high docking score and low brain-blood barrier permeability"),
    ("lung_cancer/KRAS_G12C",
     "Generate inhibitors of KRAS protein with G12C mutation. The inhibitors should be "
     "selective, meaning they should not bind with HRAS and NRAS proteins."),
    ("drug_resistance/STAT3",
     "Can you suggest molecules that inhibit STAT3 with water solubility greater than "
     "60 ug/mL and inhibitory ability to P450 CYP1A2?"),
]


def _inventory_index(inv):
    """Return {server: set(tools)} for gap checking."""
    return {srv: {name for name, _ in items} for srv, items in inv.items()}


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    index = _inventory_index(INVENTORY)
    results = []

    for case, task in TASKS:
        print(f"\n===== {case} =====")
        print(f"task: {task}")
        try:
            plan, meta = generate_plan(task, tools=INVENTORY, model=model)
        except PlanGenerationError as exc:
            print(f"  FAILED after {len(exc.errors)} attempt(s): {exc}")
            results.append({"case": case, "ok": False, "attempts": len(exc.errors), "error": str(exc)})
            continue

        ordered = plan.topological_order()
        # capability gaps: (server, tool) not present in the inventory index
        gaps = []
        for s in plan.steps:
            for ts in s.tool_servers:
                known = index.get(ts.server, set())
                for t in ts.tools:
                    if ts.server not in index or t not in known:
                        gaps.append(f"{ts.server}:{t}")
        gaps = sorted(set(gaps))

        print(f"  model={meta['model']} attempts={meta['attempts']} steps={len(plan.steps)} "
              f"order={[s.id for s in ordered]} servers={plan.required_servers()}")
        print(f"  goal: {plan.goal}")
        for s in ordered:
            arts = ",".join(a.id for a in s.expected_artifacts) or "-"
            binds = "; ".join(f"{ts.server}[{','.join(ts.tools)}]" for ts in s.tool_servers) or "-"
            print(f"   [{s.id}] deps={s.deps} -> ({arts})")
            print(f"        {s.subtask}")
            print(f"        servers: {binds}")
        if gaps:
            print(f"  capability gaps (server:tool not in inventory -> F015c/F015d/F015e): {gaps}")
        results.append({
            "case": case, "ok": True, "model": meta["model"], "attempts": meta["attempts"],
            "steps": len(plan.steps), "order": [s.id for s in ordered],
            "servers": plan.required_servers(), "gaps": gaps, "plan": plan.model_dump(),
        })

    outdir = pathlib.Path(__file__).resolve().parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    (outdir / f"r05_plans_{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    ok = sum(r["ok"] for r in results)
    avg_steps = round(sum(r.get("steps", 0) for r in results if r["ok"]) / max(ok, 1), 1)
    print(f"\n=== {ok}/{len(results)} tasks -> valid plan · avg {avg_steps} steps · "
          f"saved results/r05_plans_{stamp}.json ===")


if __name__ == "__main__":
    main()
