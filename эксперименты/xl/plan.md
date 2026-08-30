# Experiment plan · revision 1
Goal: Generate and computationally validate novel small molecules across multiple therapeutic targets (KRAS G12C, BTK, SIRT1, etc.) using in-silico modeling and docking techniques.
Hypothesis summary: Leveraging generative models and molecular docking will yield drug-like candidates with high predicted affinity and selectivity for specified targets, satisfying the constraints of blood-brain barrier penetration and reduced cross-reactivity.
Methods: Generative Adversarial Networks (GAN) for de novo molecule design, Molecular Docking (AutoDock Vina) for binding affinity estimation, In silico ADMET profiling for toxicity and drug-likeness assessment
Total duration: 165 min

## Hypotheses
- `H1`: Structural features in the KRAS G12C switch-II pocket can be exploited to design molecules that covalently bind Cys12 with >100-fold selectivity over HRAS/NRAS due to distinct pocket dynamics
- `H2`: Optimized molecular properties (MW<350, logP~2, PSA<90 Å²) will enable non-covalent BTK inhibitors with brain penetration (Kp,uu>0.3) while maintaining picomolar potency
- `H4`: Covalent warheads targeting cysteine residues in KRAS G12C can achieve irreversible inhibition while maintaining low off-target reactivity through precise electrophilicity tuning
- `H5`: Virtual screening of 10^6 compounds will identify >100 novel KRAS G12C inhibitors with sub-micromolar predicted binding affinity and favorable drug-like properties
- `H6`: Conformationally restricted scaffolds will reduce entropic penalties upon binding to KRAS G12C switch-II pocket, improving binding affinity by >10-fold compared to flexible analogs
- `H7`: Structure-activity relationship analysis across KRAS, HRAS, NRAS isoforms will identify steric and electrostatic hotspots that enable >100-fold KRAS G12C selectivity
- `H8`: FEP+ calculations on conformationally restricted analogs will quantify entropic penalties <2 kcal/mol versus >4 kcal/mol for flexible scaffolds, correlating with observed binding affinity differences
- `H9`: ML-guided SAR analysis across the focused library will identify non-linear property combinations that predict KRAS G12C selectivity with >80% accuracy

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Develop 3 small molecules targeting KRAS G12C with high selectivity and no cross-reactivity with HRAS or NRAS. | — | Covalent KRAS inhibitors (external) | Drug-likeness (QED)/maximize; Predicted Selectivity/maximize | generated_kras_molecules.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-2 | `H1` | Dock the generated KRAS G12C candidates against HRAS and NRAS to confirm the absence of cross-reactivity. | — | Generic Docking Score (method) | Binding Affinity Delta (KRAS - HRAS)/maximize [kcal/mol difference]; Binding Affinity Delta (KRAS -  | docking_report.html (report/mcp) | `fedot_mas` |
| EXP-3 | `H2` | Generate highly potent non-covalent BTK inhibitors that will have increased permeability through the blood-brain barrier | — | Ibrutinib (external) | BBB Permeability (Kp,uu)/maximize; MW < 350/maximize [fraction compliant] | btk_candidates.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-4 | `H1` | Generate inhibitors of SIRT1 to modulate lipid metabolism and improve insulin sensitivity. | — | Resveratrol (external) | Drug-likeness/maximize | sirt1_candidates.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-5 | `H1` | Generate molecules with properties of glutamate receptor antagonists for neuroprotection. | — | Memantine (external) | Neuroprotective Score (Predicted)/maximize | glutamate_antagonists.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-6 | `H1` | Generate GSK-3beta inhibitors with high activity. | — | Lithium (external) | Predicted Activity (pIC50)/maximize | gsk3b_candidates.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-7 | `H1` | Suggest molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and the abili | — | Evolocumab (external) | Bioavailability Score/maximize | pcsk9_candidates.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-8 | `H1` | Generate 2 molecules that could help in the treatment of Parkinson's disease, focusing on compounds that support the reg | — | Levodopa (external) | Safety Profile (Hallucination risk)/minimize | parkinsons_candidates.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-9 | `H1` | Generate 2 molecules that could overcome chemotherapeutic resistance in cancer treatment, specifically targeting mechani | — | Paclitaxel (external) | Toxicity to Healthy Cells (Predicted)/minimize | chemo_resistance_candidates.smi (metrics_table/mcp) | `fedot_mas` |
| EXP-10 | `H2` | Discover therapeutic agents targeting non-covalent BTK modulation to prevent multiple sclerosis progression. | — | Evobrutinib (external) | Selectivity Index (BTK vs other kinases)/maximize | btk_ms_candidates.smi (metrics_table/mcp) | `fedot_mas` |

## EXP-1 · Develop KRAS G12C selective molecules
Route: `fedot_mas`
Hypothesis: `H1` (+H4, H5, H6, H7, H8, H9)
Question: Develop 3 small molecules targeting KRAS G12C with high selectivity and no cross-reactivity with HRAS or NRAS.
Baselines: Covalent KRAS inhibitors (external)
Metrics: Drug-likeness (QED) (maximize); Predicted Selectivity (maximize)
Analysis artifacts: generated_kras_molecules.smi [metrics_table]
Task: Generate 3 small molecules targeting KRAS G12C with high selectivity.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 3 molecules generated with valid SMILES
Expected artifacts: generated_kras_molecules.smi (data)
Duration: 15 min
Warnings: none

## EXP-2 · Dock KRAS candidates vs HRAS/NRAS
Route: `fedot_mas`
Hypothesis: `H1` (+H7)
Question: Dock the generated KRAS G12C candidates against HRAS and NRAS to confirm the absence of cross-reactivity.
Baselines: Generic Docking Score (method)
Metrics: Binding Affinity Delta (KRAS - HRAS) (maximize); Binding Affinity Delta (KRAS - NRAS) (maximize)
Analysis artifacts: docking_report.html [report]
Task: Dock the generated KRAS G12C candidates against HRAS and NRAS to confirm selectivity.
MCP/tools: DockingEngine: calculate_docking
Inputs: 1
Success criteria: C1: Docking scores obtained for HRAS and NRAS
Expected artifacts: docking_report.html (report)
Duration: 30 min
Warnings: SMILES in launch_params is a placeholder; actual SMILES from EXP-1 will be used.

## EXP-3 · Generate non-covalent BTK inhibitors
Route: `fedot_mas`
Hypothesis: `H2`
Question: Generate highly potent non-covalent BTK inhibitors that will have increased permeability through the blood-brain barrier.
Baselines: Ibrutinib (external)
Metrics: BBB Permeability (Kp,uu) (maximize); MW < 350 (maximize)
Analysis artifacts: btk_candidates.smi [metrics_table]
Task: Generate highly potent non-covalent BTK inhibitors with BBB permeability.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 5 BTK candidates generated
Expected artifacts: btk_candidates.smi (data)
Duration: 15 min
Warnings: none

## EXP-4 · Generate SIRT1 inhibitors
Route: `fedot_mas`
Hypothesis: `H1`
Question: Generate inhibitors of SIRT1 to modulate lipid metabolism and improve insulin sensitivity.
Baselines: Resveratrol (external)
Metrics: Drug-likeness (maximize)
Analysis artifacts: sirt1_candidates.smi [metrics_table]
Task: Generate inhibitors of SIRT1 to modulate lipid metabolism.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 5 SIRT1 candidates generated
Expected artifacts: sirt1_candidates.smi (data)
Duration: 15 min
Warnings: none

## EXP-5 · Generate glutamate receptor antagonists
Route: `fedot_mas`
Hypothesis: `H1`
Question: Generate molecules with properties of glutamate receptor antagonists for neuroprotection.
Baselines: Memantine (external)
Metrics: Neuroprotective Score (Predicted) (maximize)
Analysis artifacts: glutamate_antagonists.smi [metrics_table]
Task: Generate molecules with properties of glutamate receptor antagonists.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 5 neuroprotective candidates generated
Expected artifacts: glutamate_antagonists.smi (data)
Duration: 15 min
Warnings: none

## EXP-6 · Generate GSK-3beta inhibitors
Route: `fedot_mas`
Hypothesis: `H1`
Question: Generate GSK-3beta inhibitors with high activity.
Baselines: Lithium (external)
Metrics: Predicted Activity (pIC50) (maximize)
Analysis artifacts: gsk3b_candidates.smi [metrics_table]
Task: Generate GSK-3beta inhibitors with high activity.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 5 GSK-3beta candidates generated
Expected artifacts: gsk3b_candidates.smi (data)
Duration: 15 min
Warnings: none

## EXP-7 · Suggest PCSK9 inhibitors
Route: `fedot_mas`
Hypothesis: `H1`
Question: Suggest molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and the ability to cross the BBB.
Baselines: Evolocumab (external)
Metrics: Bioavailability Score (maximize)
Analysis artifacts: pcsk9_candidates.smi [metrics_table]
Task: Suggest molecules that inhibit PCSK9 with BBB penetration.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 5 PCSK9 candidates generated
Expected artifacts: pcsk9_candidates.smi (data)
Duration: 15 min
Warnings: none

## EXP-8 · Generate Parkinson's disease molecules
Route: `fedot_mas`
Hypothesis: `H1`
Question: Generate 2 molecules that could help in the treatment of Parkinson's disease, focusing on compounds that support the regulation of dopamine levels and protect neurons from oxidative stress and mitochondrial dysfunction, without hallucinations, dyskinesia, or cardiovascular issues.
Baselines: Levodopa (external)
Metrics: Safety Profile (Hallucination risk) (minimize)
Analysis artifacts: parkinsons_candidates.smi [metrics_table]
Task: Generate 2 molecules for Parkinson's treatment (dopamine regulation, neuroprotection).
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 2 Parkinson's candidates generated
Expected artifacts: parkinsons_candidates.smi (data)
Duration: 15 min
Warnings: none

## EXP-9 · Generate chemoresistance molecules
Route: `fedot_mas`
Hypothesis: `H1`
Question: Generate 2 molecules that could overcome chemotherapeutic resistance in cancer treatment, specifically targeting mechanisms such as increased drug efflux, enhanced DNA repair, or apoptosis evasion, while avoiding toxicity to healthy cells.
Baselines: Paclitaxel (external)
Metrics: Toxicity to Healthy Cells (Predicted) (minimize)
Analysis artifacts: chemo_resistance_candidates.smi [metrics_table]
Task: Generate 2 molecules to overcome chemotherapeutic resistance.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 2 chemoresistance candidates generated
Expected artifacts: chemo_resistance_candidates.smi (data)
Duration: 15 min
Warnings: none

## EXP-10 · Discover non-covalent BTK modulators for MS
Route: `fedot_mas`
Hypothesis: `H2`
Question: Discover therapeutic agents targeting non-covalent BTK modulation to prevent multiple sclerosis progression.
Baselines: Evobrutinib (external)
Metrics: Selectivity Index (BTK vs other kinases) (maximize)
Analysis artifacts: btk_ms_candidates.smi [metrics_table]
Task: Discover therapeutic agents targeting non-covalent BTK modulation for multiple sclerosis.
MCP/tools: MoleculeGenerator: generate_mols
Inputs: 0
Success criteria: C1: 5 BTK MS candidates generated
Expected artifacts: btk_ms_candidates.smi (data)
Duration: 15 min
Warnings: none

## Risks
- Generative models may produce chemically unstable or synthetically inaccessible molecules.
- Docking accuracy is dependent on the quality of the receptor structures (PDB selection).
- Selectivity predictions (KRAS vs HRAS/NRAS) are in-silico estimates requiring experimental validation.
