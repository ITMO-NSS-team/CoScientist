# Experiment plan · revision 1
Goal: Execute 10 distinct molecular generation operations for specified therapeutic targets using available computational models.
Hypothesis summary: Computational generative models can produce structurally valid small molecule candidates tailored to the specific activity and property requirements of the 10 defined biological targets.
Methods: Generative Adversarial Networks (GAN), Conditional Molecular Generation
Total duration: 100 min

## Hypotheses
- `H1`: Computational generative models can produce structurally valid small molecule candidates tailored to the specific activity and property requirements of the 10 defined biological targets.

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Generate 3 small molecules targeting KRAS G12C | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | kras_g12c_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-2 | `H1` | Generate highly potent non-covalent BTK inhibitors with increased blood-brain barrier permeability | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | btk_bbb_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-3 | `H1` | Generate inhibitors of SIRT1 to modulate lipid metabolism | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | sirt1_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-4 | `H1` | Generate glutamate receptor antagonists for neuroprotection | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | glutamate_antagonists.json (metrics_table/mcp) | `fedot_mas` |
| EXP-5 | `H1` | Generate GSK-3beta inhibitors with high activity | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | gsk3b_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-6 | `H1` | Generate PCSK9 inhibitors with enhanced bioavailability | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | pcsk9_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-7 | `H1` | Generate 2 molecules for Parkinsons disease that support dopamine regulation without hallucinations | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | parkinsons_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-8 | `H1` | Generate 2 molecules that could overcome chemotherapeutic resistance while avoiding toxicity to healthy cells | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | chemoresistance_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-9 | `H1` | Generate non-covalent BTK modulators to slow multiple sclerosis progression | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | ms_btk_mols.json (metrics_table/mcp) | `fedot_mas` |
| EXP-10 | `H1` | Generate 3 JAK2 inhibitors for myeloproliferative neoplasms | None | Random SMILES sampling (method) | Structural Validity/maximize | d36e3d994404e957:generate_mols | jak2_mols.json (metrics_table/mcp) | `fedot_mas` |

## EXP-1 · Generate KRAS G12C Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-1`]
Question: Generate 3 small molecules targeting KRAS G12C
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: kras_g12c_mols.json [metrics_table]
Task: Generate 3 small molecule candidates targeting KRAS G12C.
Rationale: Directly addresses the request for KRAS G12C targeted molecules using the available fast generation tool.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=KRAS G12C inhibitor, num=3, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: kras_g12c_mols.json (data: Generated KRAS G12C molecules)
Duration: 10 min
Warnings: none

## EXP-2 · Generate BBB-penetrant BTK Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-2`]
Question: Generate highly potent non-covalent BTK inhibitors with increased blood-brain barrier permeability
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: btk_bbb_mols.json [metrics_table]
Task: Generate highly potent non-covalent BTK inhibitors with increased blood-brain barrier permeability.
Rationale: Addresses the request for non-covalent BTK inhibitors with BBB permeability.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=Highly potent non-covalent BTK inhibitor with blood-brain barrier permeability, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: btk_bbb_mols.json (data: Generated BTK inhibitors)
Duration: 10 min
Warnings: none

## EXP-3 · Generate SIRT1 Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-3`]
Question: Generate inhibitors of SIRT1 to modulate lipid metabolism
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: sirt1_mols.json [metrics_table]
Task: Generate inhibitors of SIRT1 to modulate lipid metabolism.
Rationale: Addresses the request for SIRT1 inhibitors.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=SIRT1 inhibitor to modulate lipid metabolism, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: sirt1_mols.json (data: Generated SIRT1 inhibitors)
Duration: 10 min
Warnings: none

## EXP-4 · Generate Glutamate Receptor Antagonists
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-4`]
Question: Generate glutamate receptor antagonists for neuroprotection
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: glutamate_antagonists.json [metrics_table]
Task: Generate glutamate receptor antagonists for neuroprotection.
Rationale: Addresses the request for glutamate receptor antagonists.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=Glutamate receptor antagonist for neuroprotection, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: glutamate_antagonists.json (data: Generated glutamate antagonists)
Duration: 10 min
Warnings: none

## EXP-5 · Generate GSK-3beta Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-5`]
Question: Generate GSK-3beta inhibitors with high activity
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: gsk3b_mols.json [metrics_table]
Task: Generate GSK-3beta inhibitors with high activity.
Rationale: Addresses the request for GSK-3beta inhibitors.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=GSK-3beta inhibitor with high activity, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: gsk3b_mols.json (data: Generated GSK-3beta inhibitors)
Duration: 10 min
Warnings: none

## EXP-6 · Generate PCSK9 Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-6`]
Question: Generate PCSK9 inhibitors with enhanced bioavailability
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: pcsk9_mols.json [metrics_table]
Task: Generate PCSK9 inhibitors with enhanced bioavailability.
Rationale: Addresses the request for PCSK9 inhibitors.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=PCSK9 inhibitor with enhanced bioavailability, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: pcsk9_mols.json (data: Generated PCSK9 inhibitors)
Duration: 10 min
Warnings: none

## EXP-7 · Generate Parkinsons Molecules
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-7`]
Question: Generate 2 molecules for Parkinsons disease that support dopamine regulation without hallucinations
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: parkinsons_mols.json [metrics_table]
Task: Generate 2 molecules for Parkinsons disease that support dopamine regulation without hallucinations.
Rationale: Addresses the request for Parkinsons disease molecules with specific safety profile.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=Parkinsons disease dopamine regulation support without hallucinations, num=2, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: parkinsons_mols.json (data: Generated Parkinsons molecules)
Duration: 10 min
Warnings: none

## EXP-8 · Generate Chemoresistance Molecules
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-8`]
Question: Generate 2 molecules that could overcome chemotherapeutic resistance while avoiding toxicity to healthy cells
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: chemoresistance_mols.json [metrics_table]
Task: Generate 2 molecules that could overcome chemotherapeutic resistance while avoiding toxicity to healthy cells.
Rationale: Addresses the request for chemoresistance-overcoming molecules.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=Overcome chemotherapeutic resistance avoiding toxicity to healthy cells, num=2, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: chemoresistance_mols.json (data: Generated chemoresistance molecules)
Duration: 10 min
Warnings: none

## EXP-9 · Generate MS BTK Modulators
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-9`]
Question: Generate non-covalent BTK modulators to slow multiple sclerosis progression
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: ms_btk_mols.json [metrics_table]
Task: Generate non-covalent BTK modulators to slow multiple sclerosis progression.
Rationale: Addresses the request for non-covalent BTK modulators for MS.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=Non-covalent BTK modulator to slow multiple sclerosis progression, num=10, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: ms_btk_mols.json (data: Generated MS BTK modulators)
Duration: 10 min
Warnings: none

## EXP-10 · Generate JAK2 Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-10`]
Question: Generate 3 JAK2 inhibitors for myeloproliferative neoplasms
Dataset: None — Generative task without input dataset.
Baselines: Random SMILES sampling (method)
Metrics: Structural Validity (maximize)
Analysis artifacts: jak2_mols.json [metrics_table]
Task: Generate 3 JAK2 inhibitors for myeloproliferative neoplasms.
Rationale: Addresses the request for JAK2 inhibitors.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=JAK2 inhibitor for myeloproliferative neoplasms, num=3, upload_results_to_s3=True
Inputs: none
Success criteria: C1: Output file exists
Expected artifacts: jak2_mols.json (data: Generated JAK2 inhibitors)
Duration: 10 min
Warnings: none

## Risks
- Generated molecules may not strictly conform to all requested property constraints (e.g., BBB permeability, non-covalent binding) without dedicated structural validation steps, which are outside the scope of the current generation-only request.
