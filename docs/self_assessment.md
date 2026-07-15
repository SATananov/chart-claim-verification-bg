# Self-Assessment

This self-assessment follows the scoring criteria provided for the final Deep Learning project.

## Estimated score

| Criterion | Maximum | Self-assessment |
|---|---:|---:|
| Problem statement | 10 | 9 |
| Layout | 20 | 18 |
| Code quality | 20 | 18 |
| Previous research | 10 | 9 |
| Data gathering, cleaning and formatting | 10 | 9 |
| Testing | 10 | 10 |
| Visualization | 10 | 9 |
| Communication | 10 | 9 |
| **Estimated total** | **100** | **91** |

## 1. Problem statement — 9/10

The project investigates whether a short textual claim is supported by, refuted by, or cannot be verified from a statistical chart.

This is a real-world problem because charts are frequently interpreted incorrectly or presented together with misleading claims.

Evidence:

- `notebooks/00_project_definition.ipynb`
- `README.md`
- `docs/final_project_report.md`

The score is not 10/10 because the current prototype covers a narrow statistical domain.

## 2. Layout — 18/20

The project separates notebooks, source code, tests, reports, documentation, data and model artifacts.

The notebooks follow the experimental process from problem definition and data preparation to baselines, neural models, error analysis and final evaluation.

Evidence:

- `notebooks/`
- `src/`
- `tests/`
- `reports/`
- `docs/`

The score is reduced slightly because the repository contains several experimental artifacts that make it larger than a minimal project.

## 3. Code quality — 18/20

The main functionality is organized into reusable Python modules and functions instead of being implemented only inside notebooks.

The code uses deterministic seeds, validation checks, grouped data splitting and automated tests.

Evidence:

- `src/`
- `tests/`
- `scripts/`

The score is not maximal because the project remains an educational prototype and some experimental code could be generalized further.

## 4. Previous research — 9/10

The project discusses and compares its approach with several related works:

- FEVER
- DVQA
- PlotQA
- ChartQA
- DePlot

The comparison explains how the current project differs in task scope, data size, architecture and intended use.

Evidence:

- `docs/final_project_report.md`
- `README.md`

The score is reduced because the literature review is focused and concise rather than a comprehensive academic survey.

## 5. Data gathering, cleaning and formatting — 9/10

The project uses real Eurostat unemployment data.

The workflow documents data loading, validation, chart generation, claim generation, balancing and grouped train/validation/test splitting.

The dataset contains 360 chart-claim examples with balanced classes.

Evidence:

- `notebooks/01_data_understanding.ipynb`
- data preparation modules in `src/`
- grouped split reports in `reports/`

The score is not maximal because the final dataset is small and uses templated claims.

## 6. Testing — 10/10

The project includes:

- automated unit and integrity tests;
- grouped train/validation/test separation;
- baseline comparisons;
- validation-based model selection;
- one-time evaluation on a previously untouched test set;
- a final submission audit.

Final confirmed result:

- 49 automated tests passed;
- final audit passed 15 of 15 checks.

Evidence:

- `tests/`
- `reports/final_submission_audit.md`
- `notebooks/10_final_test_evaluation.ipynb`

## 7. Visualization — 9/10

The project contains:

- generated statistical charts;
- model comparison tables and plots;
- training curves;
- confusion matrices;
- validation and final test results.

Evidence:

- executed notebooks in `notebooks/`
- evaluation files in `reports/`

The score is reduced because the chart domain and visual styles are intentionally limited.

## 8. Communication — 9/10

The project presents the problem, methodology, results, errors and limitations in a structured way.

The reported results are:

- validation accuracy: 0.8889;
- validation macro F1: 0.8885;
- final test accuracy: 0.7778;
- final test macro F1: 0.7697.

The difference between validation and final test performance is reported honestly.

Evidence:

- `README.md`
- `docs/final_project_report.md`
- `docs/submission_checklist.md`

The score is not maximal because the project would benefit from evaluation on a larger and more varied external dataset.

## Limitations acknowledged

The project does not claim universal chart understanding.

The current model was trained on:

- a small educational dataset;
- one main statistical domain;
- a limited family of chart styles;
- templated English claims.

Predictions on arbitrary charts should therefore be treated as demonstrations rather than reliable fact-checking results.

## Terms and safety

The project uses legal, non-malicious code and public statistical data. It does not access private information, execute harmful operations or contain sensitive assets requiring an NDA.
