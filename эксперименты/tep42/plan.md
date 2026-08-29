# Experiment plan · revision 2
Goal: Разработать и валидировать пайплайн генеративного дизайна мультитаргетных лигандов для нейродегенеративных заболеваний, проверив гипотезы о корреляции критериев фильтрации и влиянии качества данных.
Hypothesis summary: Проверка гипотез об ортогональности/конфликтности критериев фильтрации (H1, H2) и влиянии очистки данных (H3) в контексте генерации кандидатов для GSK-3β и родственных мишеней.
Methods: Литературный обзор и структурный анализ лигандов, Сбор и очистка данных из PubChem, Обучение моделей QSAR/QSPR (активность, токсичность), Генеративный дизайн с использованием условных VAE/GAN, Молекулярный докинг и оценка фармакокинетических свойств, Статистический анализ корреляций критериев фильтрации
Total duration: 260 min

## Hypotheses
- `H1`: Одновременная 3-критериальная фильтрация (докинг ≤ −7 к GSK-3β, предсказанная токсичность ≤ 0.3, SA-score ≤ 6) увеличивает долю жизнеспособных кандидатов минимум в 2 раза по сравнению с базовой линией
- `H2`: Для GSK-3β существует статистически значимая отрицательная корреляция (r ≤ −0.4) между docking score и синтезируемостью (SA-score)
- `H3`: Введение структурных ограничений (rotatable bonds ≤ 7, TPSA ≤ 140 Å²) при генерации позволяет получать кандидатов с SA-score ≤ 6 без существенной потери аффинности к GSK-3β (средний docking score ухудшается не более чем на 1 ккал/моль)

## Design matrix (hypothesis → experiment → data → baseline → metrics)
| Task | Hypothesis | Question | Dataset | Baselines | Metrics | Tools | Analysis artifacts | Route |
|---|---|---|---|---|---|---|---|---|
| EXP-1 | `H1` | Какие структурные классы лигандов известны для мишени и какие противоречия в данных активности существуют? | Scientific Literature | Known Inhibitors Review (prior_result) | Relevance Score/maximize | — | literature_review.md (report/research) | `research` |
| EXP-2 | `H2` | Какие структурные модификации наиболее перспективны для улучшения профиля мультитаргетных соединений? | Literature Review (EXP-1) | Standard Medicinal Chemistry Strategies (method) | Hypothesis Clarity/compare | — | hypotheses.md (report/research) | `research` |
| EXP-3 | `H3` | Позволя ли очистка данных PubChem создать репрезентативную выборку для обучения моделей активности и токсичности? | PubChem BioAssay | Raw PubChem Data (external) | Data Validity Ratio/maximize | — | dataset_clean.csv (metrics_table/coder); data_stats.json (metrics_table/coder) | `coder` |
| EXP-4 | `H3` | Достигают ли модели активности и токсичности достаточной точности на отложенной выборке? | Cleaned Dataset (EXP-3) | Random Forest Baseline (method) | ROC-AUC/maximize; R2/maximize | — | model_metrics.json (metrics_table/coder); trained_models.pkl (config/coder) | `coder` |
| EXP-5 | `H1` | Приводит ли генерация с учетом ограничений к обогащению доли жизнеспособных кандидатов? | Generated Molecules | Random Generation (method) | Viable Candidates Ratio/maximize; Correlation(Docking, SA)/compare | GenerativeModelsMCP:generate_case_mols | candidates_filtered.json (metrics_table/mcp) | `fedot_mas` |

## EXP-1 · Литературный обзор мишени
Route: `research`
Hypothesis: `H1` [Operation: `OP-1`]
Question: Какие структурные классы лигандов известны для мишени и какие противоречия в данных активности существуют?
Dataset: Scientific Literature — Поиск публикаций по GSK-3β, мультитаргетным лигандам для болезни Альцгеймера/Паркинсона.
Baselines: Known Inhibitors Review (prior_result)
Metrics: Relevance Score (maximize)
Analysis artifacts: literature_review.md [report]
Task: Провести обзор известных лигандов мишени (GSK-3β/нейродегенерация), выделить структурные классы и зафиксировать противоречия в данных об активности.
Rationale: Необходимо определить химическое пространство и ключевые фармакофоры для корректной постановки задачи генерации и интерпретации результатов (OP-1).
MCP/tools: none
Inputs: none
Success criteria: C1: Найдено и проанализировано не менее 5 релевантных статей с данными об активности и структуре.
Expected artifacts: literature_review.md (report: Обзор литературы с выделением структурных классов.)
Duration: 30 min
Warnings: none

## EXP-2 · Формулировка гипотез
Route: `research`
Hypothesis: `H2` (+H3) [Operation: `OP-2`]
Question: Какие структурные модификации наиболее перспективны для улучшения профиля мультитаргетных соединений?
Dataset: Literature Review (EXP-1) — Использование результатов EXP-1 для формулировки гипотез.
Baselines: Standard Medicinal Chemistry Strategies (method)
Metrics: Hypothesis Clarity (compare)
Analysis artifacts: hypotheses.md [report]
Task: Сформулировать и ранжировать гипотезы о структурных модификациях, улучшающих целевой профиль (аффинность, токсичность, синтезируемость).
Rationale: Определение направлений поиска и критериев для последующей генерации и фильтрации (OP-2).
MCP/tools: none
Inputs: literature_review.md [task_artifact]
Success criteria: C2: Список гипотез с ранжированием и обоснованием.
Expected artifacts: hypotheses.md (report: Список гипотез о модификациях структуры.)
Duration: 20 min
Warnings: none

## EXP-3 · Сбор и очистка данных
Route: `coder`
Hypothesis: `H3` [Operation: `OP-3`]
Question: Позволя ли очистка данных PubChem создать репрезентативную выборку для обучения моделей активности и токсичности?
Dataset: PubChem BioAssay — Сбор данных по GSK-3β и родственным мишеням.
Baselines: Raw PubChem Data (external)
Metrics: Data Validity Ratio (maximize)
Analysis artifacts: dataset_clean.csv [metrics_table]; data_stats.json [metrics_table]
Task: Собрать из PubChem обучающую выборку «структура — активность», провести очистку, нормализацию и оценку репрезентативности.
Rationale: Создание качественной эмпирической базы для обучения предсказательных моделей (OP-3).
MCP/tools: none
Inputs: none
Success criteria: C3: Создан очищенный датасет и статистика его репрезентативности.
Expected artifacts: dataset_clean.csv (data: Очищенная выборка SMILES и активность.); data_stats.json (data: Статистика распределения свойств и чистоты данных.)
Duration: 60 min
Warnings: none

## EXP-4 · Обучение предсказательных моделей
Route: `coder`
Hypothesis: `H3` [Operation: `OP-4`]
Question: Достигают ли модели активности и токсичности достаточной точности на отложенной выборке?
Dataset: Cleaned Dataset (EXP-3) — Использование dataset_clean.csv
Baselines: Random Forest Baseline (method)
Metrics: ROC-AUC (maximize); R2 (maximize)
Analysis artifacts: model_metrics.json [metrics_table]; trained_models.pkl [config]
Task: Обучить предсказательные модели активности и токсичности, провести валидацию на отложенной выборке. При недостаточном качества вернуться к очистке данных.
Rationale: Создание инструментов для виртуального скрининга сгенерированных кандидатов (OP-4).
MCP/tools: none
Inputs: dataset_clean.csv [task_artifact]
Success criteria: C4: Модель валидирована, метрики превышают пороговые значения. [ROC-AUC >= 0.7]
Expected artifacts: model_metrics.json (data: Метрики валидации моделей.); trained_models.pkl (model: Сериализованные модели активности и токсичности.)
Duration: 90 min
Warnings: none

## EXP-5 · Генерация и отбор кандидатов
Route: `fedot_mas`
Hypothesis: `H1` (+H2) [Operation: `OP-5`]
Question: Приводит ли генерация с учетом ограничений к обогащению доли жизнеспособных кандидатов?
Dataset: Generated Molecules — Результат работы генеративной модели.
Baselines: Random Generation (method)
Metrics: Viable Candidates Ratio (maximize); Correlation(Docking, SA) (compare)
Analysis artifacts: candidates_filtered.json [metrics_table]
Task: Сгенерировать молекулы-кандидаты для нейродегенеративных мишеней, отфильтровать по активности, токсичности и синтезируемости, оценить аффинность к ортологам.
Rationale: Получение итогового списка кандидатов и проверка гипотез о многокритериальной фильтрации (OP-5).
MCP/tools: GenerativeModelsMCP (http://10.32.2.2:8764/mcp): generate_case_mols
Launch params: case=alzheimer, num=100, upload_results_to_s3=True, return_inline_results=False
Inputs: none
Success criteria: C5: Получен список сгенерированных молекул.
Expected artifacts: candidates_filtered.json (data: Список кандидатов с предсказанными свойствами.)
Duration: 60 min
Warnings: Фильтрация по докингу и токсичности выполняется постфактум или внутри модели как условия.

## Risks
- Низкое качество данных PubChem может потребовать дополнительных итераций очистки (H3).
- Генеративная модель может не выдать достаточное число кандидатов с Docking <= -7, что затруднит проверку H1.
- Отсутствие прямой интеграции докинга в инструмент генерации может потребовать ручной оркестрации шагов фильтрации.
