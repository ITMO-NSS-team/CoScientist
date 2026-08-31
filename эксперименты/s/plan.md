# Experiment plan · revision 1
Goal: Identify, computationally evaluate, and plan synthesis for 3 selective KRAS G12C inhibitors.
Hypothesis summary: Covalent and non-covalent small molecules can be designed to selectively target KRAS G12C by exploiting the switch-II pocket and isoform-specific water networks, achieving >100-fold selectivity over HRAS/NRAS.
Methods: Generative molecular modeling using Conditional VAE, Structure-based molecular docking against isoform-specific targets, In silico selectivity profiling via binding energy comparison, Synthetic route planning and experimental assay design
Total duration: 135 min

## Hypotheses
- `H1`: Synthesized small molecules exhibiting high binding affinity (IC50 < 100 nM) in KRAS G12C biochemical assays will demonstrate proportional functional activity (≥50% inhibition of ERK phosphorylation at 1 μM) in cellular assays with no detectable cross-reactivity (IC50 > 10 μM) to HRAS or NRAS.
- `H2`: Specific covalent warhead chemistries (acrylamide vs. chloroacetamide) will correlate with differential assay readouts in terms of irreversible binding kinetics and sustained pathway inhibition in cellular assays.
- `H4`: Covalent small molecules targeting the switch-II pocket of KRAS G12C that exploit the unique cysteine-12 residue and isoform-specific structural differences will achieve >100-fold selectivity over HRAS and NRAS.
- `H5`: Non-covalent allosteric inhibitors targeting the KRAS G12C-specific α3-β4 interface pocket will achieve selectivity through isoform-specific conformational dynamics and sequence variations not present in HRAS or NRAS.
- `H6`: Synthesized small molecules exhibiting high binding affinity (IC50 < 100 nM) in KRAS G12C biochemical assays will demonstrate proportional functional activity (≥50% inhibition of ERK phosphorylation at 1 μM) in cellular assays with no detectable cross-reactivity (IC50 > 10 μM) to HRAS or NRAS.
- `H7`: Specific covalent warhead chemistries (acrylamide vs. chloroacetamide) will correlate with differential assay readouts in terms of irreversible binding kinetics and sustained pathway inhibition in cellular assays.
- `H8`: A standardized biochemical-to-cellular assay cascade, including a covalent adduct detection readout, achieves ≥80% predictive correlation between biochemical IC50 (KRAS G12C) and cellular p-ERK inhibition for synthesized covalent inhibitors.
- `H10`: Systematic stereochemical and electronic modulation of substituents adjacent to the covalent warhead in KRAS G12C inhibitors will identify structures that achieve >100-fold selectivity by exploiting steric clashes and distinct dynamic water networks present in HRAS/NRAS but absent in KRAS G12C.

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Identify candidate small molecules targeting KRAS G12C | KRAS G12C Case Parameters | Standard KRAS Inhibitors (prior_result) | Molecule Count/compare; Drug-likeness/maximize | d36e3d994404e957:generate_case_mols | generated_molecules.json (metrics_table/mcp) | `fedot_mas` |
| EXP-2 | `H4` | Evaluate selectivity of candidate molecules for KRAS over HRAS and NRAS | Generated Molecules from EXP-1 | Cross-docking baseline (method) | Selectivity Score (KRAS vs HRAS/NRAS)/maximize | — | selectivity_analysis.py (code/coder); docking_results.csv (metrics_table/coder) | `coder` |
| EXP-3 | `H8` | Synthesize and test the three most promising small molecules | Top 3 Selective Candidates | Standard Sotorasib Assay Protocol (method) | Protocol Completeness/maximize | — | synthesis_testing_plan.md (report/coder) | `coder` |

## EXP-1 · Generate KRAS G12C Candidates
Route: `fedot_mas`
Hypothesis: `H1` (+H4, H5, H10) [Operation: `OP-1`]
Question: Identify candidate small molecules targeting KRAS G12C
Dataset: KRAS G12C Case Parameters — Uses the 'KRAS G12C' case parameter in the CVAE model to bias generation towards the target.
Baselines: Standard KRAS Inhibitors (prior_result)
Metrics: Molecule Count (compare); Drug-likeness (maximize)
Analysis artifacts: generated_molecules.json [metrics_table]
Task: Generate a library of small molecules specifically targeting KRAS G12C using a conditional VAE tuned for therapeutic targets.
Rationale: OP-1 requires identifying candidate molecules. The 'generate_case_mols' tool is available and tuned for specific disease cases, fitting the KRAS G12C requirement.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_case_mols
Launch params: case=KRAS G12C, num=20, upload_results_to_s3=True, output_s3_prefix=kras_g12c_gen
Inputs: none
Success criteria: C1: Generate at least 10 candidate molecules [molecule_count >= 10]
Expected artifacts: generated_molecules.json (data: List of generated SMILES strings and properties.)
Duration: 30 min
Warnings: none

## EXP-2 · Evaluate Selectivity via Docking
Route: `coder`
Hypothesis: `H4` (+H10) [Operation: `OP-2`]
Question: Evaluate selectivity of candidate molecules for KRAS over HRAS and NRAS
Dataset: Generated Molecules from EXP-1 — Input SMILES strings derived from generated_molecules.json
Baselines: Cross-docking baseline (method)
Metrics: Selectivity Score (KRAS vs HRAS/NRAS) (maximize)
Analysis artifacts: selectivity_analysis.py [code]; docking_results.csv [metrics_table]
Task: Select top candidates from EXP-1 and perform molecular docking against KRAS G12C (6OIM), HRAS (4Q21), and NRAS (5UHV) to computationally assess selectivity.
Rationale: OP-2 requires evaluating selectivity over HRAS/NRAS. While 'calculate_docking' handles single PDB inputs, evaluating selectivity requires comparing scores across isoforms, necessitating a Coder task to orchestrate the tool against multiple targets.
MCP/tools: none
Launch params: pdb_kras=6OIM, pdb_hras=4Q21, pdb_nras=5UHV, top_n=5
Inputs: generated_molecules.json [task_artifact]
Success criteria: C2: Docking scores calculated for top candidates against all 3 isoforms
Expected artifacts: docking_results.csv (data: Docking scores and selectivity ratios for candidates.)
Duration: 45 min
Warnings: none

## EXP-3 · Plan Synthesis and Testing
Route: `coder`
Hypothesis: `H8` (+H2, H6, H7) [Operation: `OP-3`]
Question: Synthesize and test the three most promising small molecules
Dataset: Top 3 Selective Candidates — Structures and properties from EXP-2 results
Baselines: Standard Sotorasib Assay Protocol (method)
Metrics: Protocol Completeness (maximize)
Analysis artifacts: synthesis_testing_plan.md [report]
Task: Develop a synthesis and testing protocol for the top 3 selective candidates identified in EXP-2, including biochemical and cellular assay plans.
Rationale: OP-3 requires synthesis and testing. As no wet-lab automation tools are available, a Coder task will generate the detailed experimental protocols and analysis plan.
MCP/tools: none
Launch params: target_count=3
Inputs: docking_results.csv [task_artifact]
Success criteria: C3: Plan generated for 3 molecules including synthesis and testing steps
Expected artifacts: synthesis_testing_plan.md (report: Detailed protocols for synthesis, biochemical assay, and cellular selectivity testing.)
Duration: 60 min
Warnings: none

## Risks
- Generated molecules may not be synthetically accessible despite high computational scores.
- Computational docking scores may not accurately predict in vitro selectivity due to flexibility of RAS isoforms.
