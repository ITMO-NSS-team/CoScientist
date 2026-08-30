# Experiment plan · revision 1
Goal: Generate 3 small molecule candidates designed to selectively target KRAS G12C using computational generative methods.
Hypothesis summary: Novel heterocyclic scaffolds with electrophilic warheads can be generated to fit the KRAS G12C Switch-II pocket, achieving high affinity and selectivity over HRAS/NRAS.
Methods: Generative Adversarial Networks (GAN) for molecular design, Structure-based drug design principles
Total duration: 15 min

## Hypotheses
- `H1`: A heterocyclic scaffold with acrylamide/chloroacetamide warhead positioned to form optimal H-bonds with SIIP residues (Asp12, Thr58, Asp69) will achieve KRAS G12C docking score <-9.0 kcal/mol, predicted IC50 <100 nM, and >100-fold selectivity vs HRAS/NRAS
- `H2`: Exploiting Tyr96 bulk and H95 orientation differences in the SIIP between KRAS G12C (Y96, H95) vs HRAS/NRAS will enable scaffold designs with >100-fold selectivity through steric complementarity that accommodates KRAS but clashes with HRAS/NRAS

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Develop 3 small molecules targeting KRAS G12C with high selectivity and no cross-reactivity with HRAS or NRAS | GAN Latent Space | Sotorasib (AMG 510) (prior_result); Adagrasib (MRTX849) (prior_result) | molecule_count/maximize | d36e3d994404e957:generate_mols | kras_g12c_candidates.json (metrics_table/mcp) | `fedot_mas` |

## EXP-1 · Generate KRAS G12C Selective Inhibitors
Route: `fedot_mas`
Hypothesis: `H1` (+H2) [Operation: `OP-1`]
Question: Develop 3 small molecules targeting KRAS G12C with high selectivity and no cross-reactivity with HRAS or NRAS
Dataset: GAN Latent Space — Implicit training data within the generate_mols GAN model
Baselines: Sotorasib (AMG 510) (prior_result); Adagrasib (MRTX849) (prior_result)
Metrics: molecule_count (maximize)
Analysis artifacts: kras_g12c_candidates.json [metrics_table]
Task: Generate 3 small molecules targeting KRAS G12C with high selectivity and no cross-reactivity with HRAS or NRAS using the GAN-based molecular generator.
Rationale: The source request explicitly requires the development of 3 small molecules. The available MCP tool 'generate_mols' supports this operation via a fast GAN generator. The 'case' parameter will be utilized to bias the generation towards KRAS G12C covalent inhibitors and selectivity constraints.
MCP/tools: d36e3d994404e957 (http://10.32.2.2:8764/mcp): generate_mols
Launch params: case=Develop 3 small molecules targeting KRAS G12C with high selectivity and no cross-reactivity with HRAS or NRAS. Prioritize covalent inhibitors with acrylamide warheads targeting the Switch-II pocket., num=3, upload_results_to_s3=True, return_inline_results=True
Inputs: none
Success criteria: C1: 3 molecules generated and saved
Expected artifacts: kras_g12c_candidates.json (data: JSON output containing 3 generated SMILES strings for KRAS G12C inhibitors.)
Duration: 15 min
Warnings: Generic GAN generator may not strictly enforce steric selectivity constraints without explicit structural validation.

## Risks
- The 'generate_mols' tool is a generic GAN and may not produce molecules that strictly adhere to the specific structural constraints of the KRAS G12C Switch-II pocket or the covalent warhead positioning required for high selectivity.
- Without explicit docking validation in this plan, the predicted 'high selectivity' against HRAS/NRAS is based solely on the prompt optimization and not verified structural alignment.
