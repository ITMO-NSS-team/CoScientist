# Experiment plan · revision 1
Goal: Generate candidate small molecule inhibitors for GSK-3beta, KRAS G12C, and BTK, and candidate molecules for lipid spectrum disorder treatment using a multi-property conditional VAE.
Hypothesis summary: The multi-property conditional VAE will generate valid molecular structures that satisfy the chemical constraints and biological activity profiles for the specified targets (GSK-3beta, KRAS G12C, BTK) and the lipid disorder indication.
Methods: Multi-property conditional Variational Autoencoder (CVAE), Computational molecular generation, Target-specific property optimization
Total duration: 20 min

## Hypotheses
- `H1`: Generative modeling can produce novel small molecules with high predicted activity for GSK-3beta, KRAS G12C, BTK, and lipid regulation profiles.

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Generate GSK-3beta inhibitors with high activity | GSK-3beta Internal Knowledge | Random Sampling (method) | validity_score/maximize | d36e3d994404e957:generate_case_mols | gsk3b_molecules.json (metrics_table/mcp) | `fedot_mas` |
| EXP-2 | `H1` | Suggest some small molecules that inhibit KRAS G12C - a target responsible for non-small cell lung cancer | KRAS G12C Internal Knowledge | Random Sampling (method) | validity_score/maximize | d36e3d994404e957:generate_case_mols | kras_molecules.json (metrics_table/mcp) | `fedot_mas` |
| EXP-3 | `H1` | Generate high activity tyrosine-protein kinase BTK inhibitors | BTK Internal Knowledge | Random Sampling (method) | validity_score/maximize | d36e3d994404e957:generate_case_mols | btk_molecules.json (metrics_table/mcp) | `fedot_mas` |
| EXP-4 | `H1` | Generate 2 molecules that would help with a blood lipid spectrum disorder (raised cholesterol, triglycerides, LDL/VLDL a | Lipid Disorder Internal Knowledge | Random Sampling (method) | validity_score/maximize | d36e3d994404e957:generate_case_mols | lipid_molecules.json (metrics_table/mcp) | `fedot_mas` |

## EXP-1 · Generate GSK-3beta inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-1`]
Question: Generate GSK-3beta inhibitors with high activity
Dataset: GSK-3beta Internal Knowledge — The model uses internal weights and the 'case' parameter to guide generation.
Baselines: Random Sampling (method)
Metrics: validity_score (maximize)
Analysis artifacts: gsk3b_molecules.json [metrics_table]
Task: Generate small molecule inhibitors with high activity for GSK-3beta using the conditional VAE.
Rationale: The source request explicitly asks for GSK-3beta inhibitors. The available MCP tool 'generate_case_mols' supports molecule generation for specific cases.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_case_mols
Launch params: case=GSK-3beta, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Valid JSON output containing generated molecules
Expected artifacts: gsk3b_molecules.json (data: Generated GSK-3beta inhibitor candidates)
Duration: 5 min
Warnings: none

## EXP-2 · Generate KRAS G12C inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-2`]
Question: Suggest some small molecules that inhibit KRAS G12C - a target responsible for non-small cell lung cancer
Dataset: KRAS G12C Internal Knowledge — The model uses internal weights and the 'case' parameter to guide generation.
Baselines: Random Sampling (method)
Metrics: validity_score (maximize)
Analysis artifacts: kras_molecules.json [metrics_table]
Task: Suggest small molecules that inhibit KRAS G12C, a target for non-small cell lung cancer.
Rationale: The source request specifically targets KRAS G12C. The MCP tool is applicable for generating molecules for this specific case.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_case_mols
Launch params: case=KRAS G12C, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C2: Valid JSON output containing generated molecules
Expected artifacts: kras_molecules.json (data: Generated KRAS G12C inhibitor candidates)
Duration: 5 min
Warnings: none

## EXP-3 · Generate BTK inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-3`]
Question: Generate high activity tyrosine-protein kinase BTK inhibitors
Dataset: BTK Internal Knowledge — The model uses internal weights and the 'case' parameter to guide generation.
Baselines: Random Sampling (method)
Metrics: validity_score (maximize)
Analysis artifacts: btk_molecules.json [metrics_table]
Task: Generate high activity tyrosine-protein kinase BTK inhibitors.
Rationale: The source request requires BTK inhibitors. The MCP tool is capable of generating molecules for this target.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_case_mols
Launch params: case=BTK, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C3: Valid JSON output containing generated molecules
Expected artifacts: btk_molecules.json (data: Generated BTK inhibitor candidates)
Duration: 5 min
Warnings: none

## EXP-4 · Generate lipid disorder molecules
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-4`]
Question: Generate 2 molecules that would help with a blood lipid spectrum disorder (raised cholesterol, triglycerides, LDL/VLDL and lowered HDL), without muscle pain or liver side effects.
Dataset: Lipid Disorder Internal Knowledge — The model uses internal weights and the 'case' parameter to guide generation.
Baselines: Random Sampling (method)
Metrics: validity_score (maximize)
Analysis artifacts: lipid_molecules.json [metrics_table]
Task: Generate 2 molecules for blood lipid spectrum disorder (raised cholesterol, triglycerides, LDL/VLDL and lowered HDL), avoiding muscle pain or liver side effects.
Rationale: The source request asks for specific molecules for lipid spectrum disorder. The MCP tool can generate molecules conditioned on this case.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_case_mols
Launch params: case=blood lipid spectrum disorder, num=2, upload_results_to_s3=True
Inputs: none
Success criteria: C4: Valid JSON output containing 2 generated molecules
Expected artifacts: lipid_molecules.json (data: Generated lipid disorder treatment candidates)
Duration: 5 min
Warnings: none

## Risks
- The 'generate_case_mols' tool uses a 'HARDCODED disease case' model. If the specific target strings (GSK-3beta, KRAS G12C, BTK, blood lipid spectrum disorder) do not match the internal hardcoded cases, the generation may fail or return irrelevant results.
- The model generates molecules based on learned distributions but does not guarantee wet-lab activity or absence of side effects (e.g., muscle pain, liver toxicity) without further validation.
