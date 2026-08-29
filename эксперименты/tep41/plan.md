# Experiment plan · revision 2
Goal: Создать токсикологический профиль метаболитов Heracleum sosnowskyi, включая кластеризацию, предсказание LD50, оценку области применимости (AD) и специфической токсичности (гепато-, кардио-, канцерогенность) для определения фармацевтического потенциала.
Hypothesis summary: Опубликованные данные покрывают значительную часть биоактивных соединений (гипотезы H4, H9), а набор из 20+ подтверждённых соединений достаточен для построения надежных моделей токсичности (гипотезы H5, H10).
Methods: Systematic literature review (OpenAlex, PubMed), Data curation (SMILES standardization, deduplication), Molecular clustering (structural similarity fingerprints), In-silico toxicology prediction (LD50, AD, endpoints), Statistical analysis of applicability domain, Cluster-based toxicity profiling
Total duration: 240 min

## Hypotheses
- `H4`: Published data covers 70%+ of major bioactive compounds
- `H5`: 20+ experimentally confirmed compounds sufficient for modeling
- `H9`: Published data covers 70%+ of major bioactive compounds
- `H10`: 20+ experimentally confirmed compounds sufficient for modeling

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H4` | Какие метаболиты борщевика Сосновского подтверждены экспериментально в литературе? | Scientific Literature Corpus | Manual expert review (prior_result) | Unique compounds identified/maximize | — | literature_review.md (report/research) | `research` |
| EXP-2 | `H5` | Достаточно ли данных (20+ соединений) для моделирования и покрывают ли они 70% состава? | Curated H. sosnowskyi Metabolites | Raw literature extraction (method) | Validated unique structures/maximize | — | compounds.json (metrics_table/coder) | `coder` |
| EXP-3 | `H5` | На какие кластеры разбиваются метаболиты борщевика по молекулярному сходству? | Curated H. sosnowskyi Metabolites | Random assignment (method) | Silhouette score/maximize | — | clusters.json (metrics_table/coder) | `coder` |
| EXP-4 | `H5` | Каковы предсказанные значения LD₅₀ для метаболитов? | Curated H. sosnowskyi Metabolites | Chemical similarity read-across (method) | Prediction coverage/maximize | bfc62a287aaf7b5a:predict_molecule_profile | toxicity_profiles.json (metrics_table/mcp) | `fedot_mas` |
| EXP-5 | `H1` | Какова область применимости моделей и надёжность предсказаний? | Toxicity Predictions | Global AD (method) | Compounds within AD/maximize | — | ad_analysis.json (metrics_table/coder) | `coder` |
| EXP-6 | `H1` | Какие классы соединений обладают наивысшей специфической токсичностью? | Toxicity Predictions + Clusters | LD50 only assessment (method) | Risk categories defined/compare | — | specific_toxicity_report.md (report/coder) | `coder` |

## EXP-1 · Literature Review: H. sosnowskyi Composition
Route: `research`
Hypothesis: `H4` (+H9) [Operation: `OP-1`]
Question: Какие метаболиты борщевика Сосновского подтверждены экспериментально в литературе?
Dataset: Scientific Literature Corpus — OpenAlex and internal database
Baselines: Manual expert review (prior_result)
Metrics: Unique compounds identified (maximize)
Analysis artifacts: literature_review.md [report]
Task: Conduct a systematic review of publications to identify chemical compounds found in Heracleum sosnowskyi.
Rationale: Primary step to gather experimental data on metabolite presence (OP-1) and test hypothesis H4/H9 regarding data coverage.
MCP/tools: none
Inputs: none
Success criteria: C1: Literature review compiled with list of compounds
Expected artifacts: literature_review.md (report: Summary of chemical composition data sources)
Duration: 60 min
Warnings: none

## EXP-2 · Data Curation: Selection and Standardization
Route: `coder`
Hypothesis: `H5` (+H4, H9, H10) [Operation: `OP-2`]
Question: Достаточно ли данных (20+ соединений) для моделирования и покрывают ли они 70% состава?
Dataset: Curated H. sosnowskyi Metabolites — Derived from EXP-1
Baselines: Raw literature extraction (method)
Metrics: Validated unique structures (maximize)
Analysis artifacts: compounds.json [metrics_table]
Task: Process literature findings to select compounds, standardize structures (SMILES), deduplicate, and mark experimental/predicted status.
Rationale: Creates the structured dataset needed for modeling (OP-2) and validates hypotheses H4, H5, H9, H10 regarding data coverage and sufficiency.
MCP/tools: none
Inputs: literature_review.md [task_artifact]
Success criteria: C2: Standardized dataset with >20 compounds [Compound count >= 20]
Expected artifacts: compounds.json (data: List of standardized SMILES with metadata)
Duration: 40 min
Warnings: none

## EXP-3 · Clustering by Molecular Similarity
Route: `coder`
Hypothesis: `H5` (+H10) [Operation: `OP-3`]
Question: На какие кластеры разбиваются метаболиты борщевика по молекулярному сходству?
Dataset: Curated H. sosnowskyi Metabolites — Input from EXP-2
Baselines: Random assignment (method)
Metrics: Silhouette score (maximize)
Analysis artifacts: clusters.json [metrics_table]
Task: Perform molecular clustering (e.g. using fingerprints) to group metabolites and interpret chemical classes.
Rationale: Structural analysis step (OP-3) to identify chemical classes and support interpretation of toxicity patterns. Tests H5/H10 sufficiency by ensuring the set allows meaningful clustering.
MCP/tools: none
Inputs: compounds.json [task_artifact]
Success criteria: C3: Cluster assignments generated
Expected artifacts: clusters.json (data: Cluster labels for each compound)
Duration: 30 min
Warnings: none

## EXP-4 · Toxicity Prediction and Modeling (LD50)
Route: `fedot_mas`
Hypothesis: `H5` (+H10) [Operation: `OP-4`]
Question: Каковы предсказанные значения LD₅₀ для метаболитов?
Dataset: Curated H. sosnowskyi Metabolites — Input from EXP-2
Baselines: Chemical similarity read-across (method)
Metrics: Prediction coverage (maximize)
Analysis artifacts: toxicity_profiles.json [metrics_table]
Task: Predict LD50 values (6 routes) using in-silico models for all curated compounds.
Rationale: Core computational step (OP-4) to fill data gaps. Validates H5/H10 by generating model outputs.
MCP/tools: bfc62a287aaf7b5a (http://10.32.11.45:7336/mcp): predict_molecule_profile
Launch params: upload_results_to_s3=True
Inputs: compounds.json [task_artifact]
Success criteria: C4: LD50 predictions generated for compounds
Expected artifacts: toxicity_profiles.json (data: Full toxicity profile including LD50 and endpoints)
Duration: 60 min
Warnings: none

## EXP-5 · Applicability Domain (AD) Analysis
Route: `coder`
Hypothesis: `H1` [Operation: `OP-5`]
Question: Какова область применимости моделей и надёжность предсказаний?
Dataset: Toxicity Predictions — Output from EXP-4
Baselines: Global AD (method)
Metrics: Compounds within AD (maximize)
Analysis artifacts: ad_analysis.json [metrics_table]
Task: Quantify the reliability of predictions by analyzing the Applicability Domain (AD) for each compound and cluster.
Rationale: Assesses model reliability (OP-5) as required for robust conclusions.
MCP/tools: none
Inputs: toxicity_profiles.json [task_artifact]
Success criteria: C5: AD metrics calculated
Expected artifacts: ad_analysis.json (data: AD scores per compound and cluster)
Duration: 20 min
Warnings: none

## EXP-6 · Specific Toxicity Profiling
Route: `coder`
Hypothesis: `H1` [Operation: `OP-6`]
Question: Какие классы соединений обладают наивысшей специфической токсичностью?
Dataset: Toxicity Predictions + Clusters — Combined from EXP-3 and EXP-4
Baselines: LD50 only assessment (method)
Metrics: Risk categories defined (compare)
Analysis artifacts: specific_toxicity_report.md [report]
Task: Analyze specific endpoints (hepatotoxicity, DILI, cardiotoxicity, carcinogenicity) for the most toxic clusters.
Rationale: Final analysis step (OP-6) to characterize risks of the most dangerous chemical groups.
MCP/tools: none
Inputs: clusters.json [task_artifact]; toxicity_profiles.json [task_artifact]; ad_analysis.json [task_artifact]
Success criteria: C6: Toxic clusters identified with endpoints
Expected artifacts: specific_toxicity_report.md (report: Analysis of hepatotoxicity, DILI, etc.)
Duration: 30 min
Warnings: none

## Risks
- Experimental data in literature may be fragmented, affecting hypothesis H4/H9 confirmation
- In-silico models may have limited accuracy for novel derivatives specific to H. sosnowskyi
- AD analysis might reveal low confidence for a significant subset of compounds
