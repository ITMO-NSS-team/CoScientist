# Experiment plan · revision 2
Goal: Generate a list of candidate molecules optimized for KRAS G12C affinity and drug-like properties.
Hypothesis summary: Optimized drug-like properties predict high KRAS G12C docking affinity
Methods: Generative molecular modeling using GAN, In-silico property filtering
Total duration: 30 min

## Hypotheses
- `H1`: Optimized drug-like properties predict high KRAS G12C docking affinity

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Can we generate drug-like molecules with potential high affinity for KRAS G12C? | None | Random Sampling (method) | Drug-likeness (QED)/maximize; Synthesizability Score/maximize | d36e3d994404e957:generate_mols | generated_molecules.json (metrics_table/mcp) | `fedot_mas` |

## EXP-1 · Generate KRAS G12C Candidate Molecules
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-1`]
Question: Can we generate drug-like molecules with potential high affinity for KRAS G12C?
Dataset: None — De novo generation from latent space.
Baselines: Random Sampling (method)
Metrics: Drug-likeness (QED) (maximize); Synthesizability Score (maximize)
Analysis artifacts: generated_molecules.json [metrics_table]
Task: Generate a set of novel molecules targeting KRAS G12C with optimized drug-like properties using a GAN-based generator.
Rationale: The primary operation (OP-1) requires the generation of candidate molecules. This task utilizes the available generate_mols capability to produce a list of structures that serve as the foundation for further analysis.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=KRAS G12C inhibitor, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Successfully generated candidate molecules file
Expected artifacts: generated_molecules.json (data: List of generated candidate molecules with SMILES strings.)
Duration: 30 min
Warnings: none

## Risks
- Generated molecules may require further docking validation which is not covered in this single-task plan due to operation constraints.
- The generator may produce molecules that are chemically unstable or difficult to synthesize despite optimization heuristics.
