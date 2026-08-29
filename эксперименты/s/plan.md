# Experiment plan · revision 2
Goal: Design and validate 3 small molecule candidates that selectively bind to KRAS G12C with a predicted binding energy < -50 kJ/mol (covalent) or < -45 kJ/mol (non-covalent) and >80% selectivity over HRAS/NRAS.
Hypothesis summary: Covalent and non-covalent inhibitors targeting specific pockets of KRAS G12C can achieve high selectivity and binding affinity.
Methods: Structure-Based Drug Design (SBDD), Molecular Docking, Covalent Docking Simulation, Selectivity Analysis
Total duration: 90 min

## Hypotheses
- `H1`: Covalent inhibitors with electrophilic warheads targeting cysteine-12 in the KRAS G12C switch-II pocket will achieve >80% selectivity over HRAS/NRAS due to absence of reactive cysteine at position 12 in other isoforms
- `H2`: Non-covalent inhibitors designed to exploit differential residue patterns in the switch-II pocket (particularly differences at positions 12, 13, 95, 96) between KRAS G12C and HRAS/NRAS will achieve >80% selectivity without requiring covalent warheads

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Can 3 novel small molecules be designed that achieve a predicted binding energy < -50 kJ/mol for KRAS G12C with >80% sel | KRAS_G12C_Structures | Sotorasib (AMG 510) (prior_result); Adagrasib (MRTX849) (prior_result) | Predicted Binding Energy (kJ/mol)/minimize [Covalent binding energy < -50 kJ/mol]; Selectivity Perce | kras_design_pipeline.py (code/coder); molecule_results.json (metrics_table/coder) | `coder` |

## EXP-1 · KRAS G12C Molecule Design and Validation
Route: `coder`
Hypothesis: `H1` (+H2)
Question: Can 3 novel small molecules be designed that achieve a predicted binding energy < -50 kJ/mol for KRAS G12C with >80% selectivity over HRAS/NRAS?
Dataset: KRAS_G12C_Structures — Publicly available PDB structures for KRAS G12C (e.g., 6OIM), HRAS (e.g., 5P21), and NRAS (e.g., 5USJ) for docking simulations.
Baselines: Sotorasib (AMG 510) (prior_result); Adagrasib (MRTX849) (prior_result)
Metrics: Predicted Binding Energy (kJ/mol) (minimize); Selectivity Percentage (%) (maximize)
Analysis artifacts: kras_design_pipeline.py [code]; molecule_results.json [metrics_table]
Task: Design 3 small molecules targeting KRAS G12C and computationally validate their binding affinity and selectivity against HRAS and NRAS using molecular docking techniques.
MCP/tools: none
Inputs: 0
Success criteria: C1: 3 molecules generated with required properties
Expected artifacts: kras_design_pipeline.py (code); molecule_results.json (data)
Duration: 90 min
Warnings: none

## Risks
- Designed molecules may violate Lipinski's Rule of Five despite high binding affinity.
- Computational docking scores may not perfectly correlate with in vitro binding affinity.
