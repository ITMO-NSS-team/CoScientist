#!/usr/bin/env python
"""R05 benchmark — generate structured plans for a few dataset_S tasks (F015a).

Exercises CoScientist.experiments.generate_plan end-to-end (real LLM, strict JSON,
validate-then-repair) on representative drug-design tasks from dataset_S.xlsx, then
checks each plan is a valid DAG and reports capability gaps (tools the planner named
that aren't in the inventory — the future F015c "build it" branch).

Usage: python scripts/experiments/r05_plan_benchmark.py [model]
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
from CoScientist.experiments import PlanGenerationError, generate_plan  # noqa: E402

# Representative stand-in for the LIVE MCP index (roadmap R07 replaces this).
# Some names may not exist as real tools yet — that is the point: F015c detects gaps.
INVENTORY = [
    ("generate_case_mols", "generate candidate molecules for a known target case (GAN/Transformer)"),
    ("generate_mols", "sample molecules from a trained generative model"),
    ("calculate_docking", "docking score of a SMILES against a protein (pdb id)"),
    ("get_rdkit_properties", "RDKit physchem properties from SMILES (MW, logP, TPSA, QED)"),
    ("predict_bbb_permeability", "predict blood-brain-barrier permeability from SMILES"),
    ("predict_admet", "predict ADMET / toxicity endpoints from SMILES"),
    ("retrosynthesis_route", "retrosynthetic route search for a SMILES"),
    ("fetch_protein_activities", "fetch known actives for a protein from ChEMBL/BindingDB"),
]

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


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    inv_names = {n for n, _ in INVENTORY}
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
        used = plan.all_required_tools()
        gaps = [t for t in used if t not in inv_names]
        print(f"  model={meta['model']} attempts={meta['attempts']} steps={len(plan.steps)} "
              f"order={[s.id for s in ordered]}")
        print(f"  goal: {plan.goal}")
        for s in ordered:
            arts = ",".join(a.id for a in s.expected_artifacts) or "-"
            print(f"   [{s.id}] deps={s.deps} tools={s.required_tools} -> ({arts})")
            print(f"        {s.subtask}")
        if gaps:
            print(f"  capability gaps (not in inventory -> F015c/F015d/F015e): {sorted(set(gaps))}")
        results.append({
            "case": case, "ok": True, "model": meta["model"], "attempts": meta["attempts"],
            "steps": len(plan.steps), "order": [s.id for s in ordered],
            "gaps": sorted(set(gaps)), "plan": plan.model_dump(),
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
