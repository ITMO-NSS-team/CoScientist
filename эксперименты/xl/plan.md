# Experiment plan · revision 2
Goal: Generate and validate in silico small molecule candidates for multiple therapeutic targets (KRAS G12C, BTK, SIRT1, Glutamate Receptors, GSK-3beta, PCSK9, Parkinson's, Chemoresistance, MS) focusing on selectivity, BBB permeability, and potency.
Hypothesis summary: Structure-based design and conditional generative models can produce novel small molecules that meet specific affinity, selectivity, and ADMET criteria for diverse oncological and neurological targets.
Methods: de novo molecular generation using conditional VAE/GAN models, structure-based drug design (SBDD) using molecular docking, in silico ADMET and toxicity profiling, selectivity screening via cross-docking
Total duration: 105 min

## Hypotheses
- `H1`: Structure-based design of 3 KRAS G12C covalent inhibitors exploiting the switch-II pocket (SIIP) with >80-fold selectivity vs HRAS/NRAS through interactions unique to G12C conformation (Cys12, His95, Arg68 network)
- `H2`: Non-covalent BTK inhibitors optimized for BBB penetration (CNS MPO > 4, PSA < 70 Å²) with >10-fold selectivity over EGFR, ITK, TEC kinases through differential hinge-binding motifs
- `H4`: Subtype-selective glutamate receptor antagonists targeting NMDA (ifenprodil-like GluN1/GluN2B interface) or AMPA (peripheral tetramerization domain) to achieve neuroprotection without excitotoxicity through partial allosteric modulation
- `H5`: GSK-3β inhibitors (IC50 < 10 nM) with >50-fold selectivity over CDK2, CDK5, GSK-3α through exploiting unique GSK-3β Arg141 pocket and hinge region (Asp133, Val135) with type I/II ATP-competitive scaffolds
- `H7`: Parkinson's disease modulators targeting dopamine D1/D3 receptors with biased agonism (G-protein over β-arrestin pathway) to provide symptomatic relief without hallucinations/dyskinesia, combined with MAO-B inhibition (≤1 nM) for neuroprotection without cardiovascular effects
- `H8`: Chemoresistance reversal agents either (a) inhibiting efflux pumps (P-gp, BCRP) with substrate-competitive binding (IC50 < 200 nM) or (b) modulating DNA repair (PARP1/ATM pathways) to sensitize resistant cancer cells to apoptosis
- `H10`: Structure-based design of 3 KRAS G12C covalent inhibitors exploiting the switch-II pocket (SIIP) with >80-fold selectivity vs HRAS/NRAS through interactions unique to G12C conformation (Cys12, His95, Arg68 network)
- `H11`: Non-covalent BTK inhibitors optimized for BBB penetration (CNS MPO > 4, PSA < 70 Å²) with >10-fold selectivity over EGFR, ITK, TEC kinases through differential hinge-binding motifs

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Can we generate 3 covalent KRAS G12C inhibitors with predicted high selectivity? | KRAS_G12C_Target | Sotorasib (prior_result) | Selectivity_Score/maximize | d36e3d994404e957:generate_mols | kras_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-2 | `H1` | Do KRAS candidates show weak binding to HRAS (off-target)? | HRAS_Structure | Weak_Binding (method) | Docking_Score/minimize [Docking score > -7.0 kcal/mol indicates low affinity] | bfd3f80438ba403b:calculate_docking | hras_docking_results.html (metrics_table/mcp) | `fedot_mas` |
| EXP-3 | `H2` | Can we generate non-covalent BTK inhibitors with CNS MPO > 4? | BTK_Target | Fenebrutinib (prior_result) | CNS_MPO/maximize | d36e3d994404e957:generate_mols | btk_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-4 | `H1` | Can we generate SIRT1 inhibitors with suitable potency? | SIRT1_Target | EX-527 (prior_result) | Potency_Prediction/maximize | d36e3d994404e957:generate_mols | sirt1_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-5 | `H4` | Can we generate glutamate antagonists avoiding excitotoxicity? | Glutamate_Target | Memantine (prior_result) | Neuroprotection_Score/maximize | d36e3d994404e957:generate_mols | glutamate_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-6 | `H5` | Can we generate GSK-3beta inhibitors with >50x selectivity over CDKs? | GSK3beta_Target | Tideglusib (prior_result) | Selectivity_Ratio/maximize | d36e3d994404e957:generate_mols | gsk3b_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-7 | `H1` | Can we generate small molecule PCSK9 inhibitors with BBB permeability? | PCSK9_Target | Evolocumab (prior_result) | Bioavailability_Score/maximize | d36e3d994404e957:generate_mols | pcsk9_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-8 | `H7` | Can we generate PD modulators that avoid hallucinations/dyskinesia? | PD_Targets | Pramipexole (prior_result) | Safety_Profile/maximize | d36e3d994404e957:generate_mols | pd_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-9 | `H8` | Can we generate resistance reversal agents with low toxicity? | Chemo_Targets | Venetoclax (prior_result) | Selectivity_Index/maximize | d36e3d994404e957:generate_mols | chemo_candidates.json (metrics_table/mcp) | `fedot_mas` |
| EXP-10 | `H2` | Can we generate BTK modulators suitable for MS progression prevention? | MS_BTK_Target | Tolebrutinib (prior_result) | Efficacy_Prediction/maximize | d36e3d994404e957:generate_mols | ms_btk_candidates.json (metrics_table/mcp) | `fedot_mas` |

## EXP-1 · Generate KRAS G12C Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` (+H10) [Operation: `OP-1`]
Question: Can we generate 3 covalent KRAS G12C inhibitors with predicted high selectivity?
Dataset: KRAS_G12C_Target — Target context: KRAS G12C switch-II pocket (SIIP).
Baselines: Sotorasib (prior_result)
Metrics: Selectivity_Score (maximize)
Analysis artifacts: kras_candidates.json [metrics_table]
Task: Generate 3 small molecule covalent inhibitors targeting KRAS G12C with high selectivity profile.
Rationale: Fulfills OP-1 to provide candidate molecules for selectivity validation.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=3, case=KRAS_G12C, upload_results_to_s3=True
Inputs: none
Success criteria: C1: 3 molecules generated
Expected artifacts: kras_candidates.json (data: Generated KRAS G12C candidate SMILES)
Duration: 10 min
Warnings: none

## EXP-2 · Dock KRAS Candidates vs HRAS
Route: `fedot_mas`
Hypothesis: `H1` (+H10) [Operation: `OP-2`]
Question: Do KRAS candidates show weak binding to HRAS (off-target)?
Dataset: HRAS_Structure — PDB ID: 5W22 (HRAS WT).
Baselines: Weak_Binding (method)
Metrics: Docking_Score (minimize)
Analysis artifacts: hras_docking_results.html [metrics_table]
Task: Dock generated KRAS G12C candidates against HRAS to confirm absence of cross-reactivity.
Rationale: Fulfills OP-2 requirement for cross-reactivity validation against HRAS.
MCP/tools: bfd3f80438ba403b (http://10.32.11.45:7332/mcp): calculate_docking
Launch params: smiles=CC(C)CC1=NC=C(C=C1)C#N, pdb_id=5W22
Inputs: kras_candidates.json [task_artifact]
Success criteria: C2: Docking completed for HRAS
Expected artifacts: hras_docking_results.html (data: Docking visualization and scores)
Duration: 15 min
Warnings: none

## EXP-3 · Generate Non-covalent BTK Inhibitors (BBB)
Route: `fedot_mas`
Hypothesis: `H2` (+H11) [Operation: `OP-3`]
Question: Can we generate non-covalent BTK inhibitors with CNS MPO > 4?
Dataset: BTK_Target — Non-covalent BTK binding site, BBB constraints.
Baselines: Fenebrutinib (prior_result)
Metrics: CNS_MPO (maximize)
Analysis artifacts: btk_candidates.json [metrics_table]
Task: Generate highly potent non-covalent BTK inhibitors optimized for BBB permeability.
Rationale: Fulfills OP-3 to discover brain-penetrant BTK modulators.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=1, case=BTK_noncovalent_BBB, upload_results_to_s3=True
Inputs: none
Success criteria: C3: BTK candidate generated
Expected artifacts: btk_candidates.json (data: Generated BTK inhibitor SMILES)
Duration: 10 min
Warnings: none

## EXP-4 · Generate SIRT1 Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-4`]
Question: Can we generate SIRT1 inhibitors with suitable potency?
Dataset: SIRT1_Target — SIRT1 enzymatic pocket.
Baselines: EX-527 (prior_result)
Metrics: Potency_Prediction (maximize)
Analysis artifacts: sirt1_candidates.json [metrics_table]
Task: Generate SIRT1 inhibitors to modulate lipid metabolism and improve insulin sensitivity.
Rationale: Fulfills OP-4 for metabolic disease targets.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=1, case=SIRT1_inhibitor, upload_results_to_s3=True
Inputs: none
Success criteria: C4: SIRT1 candidate generated
Expected artifacts: sirt1_candidates.json (data: Generated SIRT1 inhibitor SMILES)
Duration: 10 min
Warnings: none

## EXP-5 · Generate Glutamate Receptor Antagonists
Route: `fedot_mas`
Hypothesis: `H4` [Operation: `OP-5`]
Question: Can we generate glutamate antagonists avoiding excitotoxicity?
Dataset: Glutamate_Target — NMDA/AMPA receptor sites.
Baselines: Memantine (prior_result)
Metrics: Neuroprotection_Score (maximize)
Analysis artifacts: glutamate_candidates.json [metrics_table]
Task: Generate molecules with properties of glutamate receptor antagonists for neuroprotection.
Rationale: Fulfills OP-5 for neuroprotection targets.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=1, case=Glutamate_antagonist, upload_results_to_s3=True
Inputs: none
Success criteria: C5: Glutamate antagonist generated
Expected artifacts: glutamate_candidates.json (data: Generated glutamate antagonist SMILES)
Duration: 10 min
Warnings: none

## EXP-6 · Generate GSK-3beta Inhibitors
Route: `fedot_mas`
Hypothesis: `H5` [Operation: `OP-6`]
Question: Can we generate GSK-3beta inhibitors with >50x selectivity over CDKs?
Dataset: GSK3beta_Target — ATP-competitive binding site.
Baselines: Tideglusib (prior_result)
Metrics: Selectivity_Ratio (maximize)
Analysis artifacts: gsk3b_candidates.json [metrics_table]
Task: Generate GSK-3beta inhibitors with high activity.
Rationale: Fulfills OP-6 for high-activity kinase inhibitors.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=1, case=GSK3beta_inhibitor, upload_results_to_s3=True
Inputs: none
Success criteria: C6: GSK-3beta inhibitor generated
Expected artifacts: gsk3b_candidates.json (data: Generated GSK-3beta inhibitor SMILES)
Duration: 10 min
Warnings: none

## EXP-7 · Generate PCSK9 Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` [Operation: `OP-7`]
Question: Can we generate small molecule PCSK9 inhibitors with BBB permeability?
Dataset: PCSK9_Target — PCSK9-LDLR interaction interface.
Baselines: Evolocumab (prior_result)
Metrics: Bioavailability_Score (maximize)
Analysis artifacts: pcsk9_candidates.json [metrics_table]
Task: Suggest molecules that inhibit PCSK9 with enhanced bioavailability and BBB crossing.
Rationale: Fulfills OP-7 for lipid metabolism targets.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=1, case=PCSK9_inhibitor_BBB, upload_results_to_s3=True
Inputs: none
Success criteria: C7: PCSK9 candidate generated
Expected artifacts: pcsk9_candidates.json (data: Generated PCSK9 inhibitor SMILES)
Duration: 10 min
Warnings: none

## EXP-8 · Generate Parkinson's Disease Therapeutics
Route: `fedot_mas`
Hypothesis: `H7` [Operation: `OP-8`]
Question: Can we generate PD modulators that avoid hallucinations/dyskinesia?
Dataset: PD_Targets — Dopamine D1/D3 receptors, MAO-B.
Baselines: Pramipexole (prior_result)
Metrics: Safety_Profile (maximize)
Analysis artifacts: pd_candidates.json [metrics_table]
Task: Generate 2 molecules for Parkinson's disease focusing on dopamine regulation and neuroprotection.
Rationale: Fulfills OP-8 for specific PD multi-target ligands.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=2, case=Parkinsons_disease, upload_results_to_s3=True
Inputs: none
Success criteria: C8: 2 PD candidates generated
Expected artifacts: pd_candidates.json (data: Generated PD therapeutics SMILES)
Duration: 10 min
Warnings: none

## EXP-9 · Generate Chemoresistance Reversal Agents
Route: `fedot_mas`
Hypothesis: `H8` [Operation: `OP-9`]
Question: Can we generate resistance reversal agents with low toxicity?
Dataset: Chemo_Targets — P-gp, BCRP, PARP1 pathways.
Baselines: Venetoclax (prior_result)
Metrics: Selectivity_Index (maximize)
Analysis artifacts: chemo_candidates.json [metrics_table]
Task: Generate 2 molecules to overcome chemotherapeutic resistance (efflux/DNA repair/apoptosis).
Rationale: Fulfills OP-9 for oncology resistance mechanisms.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=2, case=Chemoresistance_reversal, upload_results_to_s3=True
Inputs: none
Success criteria: C9: 2 Chemo candidates generated
Expected artifacts: chemo_candidates.json (data: Generated chemoresistance agents SMILES)
Duration: 10 min
Warnings: none

## EXP-10 · Discover MS BTK Modulators
Route: `fedot_mas`
Hypothesis: `H2` (+H11) [Operation: `OP-10`]
Question: Can we generate BTK modulators suitable for MS progression prevention?
Dataset: MS_BTK_Target — Non-covalent BTK modulation for MS.
Baselines: Tolebrutinib (prior_result)
Metrics: Efficacy_Prediction (maximize)
Analysis artifacts: ms_btk_candidates.json [metrics_table]
Task: Discover therapeutic agents targeting non-covalent BTK modulation to prevent multiple sclerosis progression.
Rationale: Fulfills OP-10 for Multiple sclerosis specific BTK modulation.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: num=1, case=Multiple_Sclerosis_BTK, upload_results_to_s3=True
Inputs: none
Success criteria: C10: MS BTK modulator generated
Expected artifacts: ms_btk_candidates.json (data: Generated MS BTK modulator SMILES)
Duration: 10 min
Warnings: none

## Risks
- Docking tool requires single SMILES/PDB; automation of candidate iteration may need orchestration support.
- Generic molecule generator may not capture specific disease nuances (e.g., MS, PD) as accurately as specialized models.
