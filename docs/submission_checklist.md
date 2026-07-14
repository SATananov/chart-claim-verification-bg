# Submission Checklist

This checklist maps the completed repository to the project assessment
criteria. It identifies evidence; it does not predict a grade.

## Repository state

- [x] Public GitHub repository
- [x] Numbered notebook workflow
- [x] Reusable Python modules in `src/`
- [x] Automated tests in `tests/`
- [x] Stored experiment reports
- [x] Final README
- [x] Final project report
- [x] Untouched test evaluation completed once
- [x] Final test completion marker committed

## Assessment evidence

| Criterion | Repository evidence |
|---|---|
| Problem statement | `README.md`, `notebooks/00_project_definition.ipynb`, `docs/final_project_report.md` |
| Project layout | `data/`, `docs/`, `notebooks/`, `reports/`, `src/`, `tests/` |
| Code quality | Reusable modules, type hints, validation checks, deterministic seeds |
| Previous research | Final report sections on FEVER, DVQA, PlotQA, ChartQA and DePlot |
| Data | Source documentation, processed CSV files, balanced labels and grouped splits |
| Testing | `python -m pytest`; confirmed state before Step 013: 38 passed |
| Visualization | Generated charts, training curves, model comparison charts and confusion matrices in executed notebooks |
| Communication | README, final report, methodology documents and saved CSV/JSON reports |

## Final metrics

| Metric | Validation | Final test |
|---|---:|---:|
| Accuracy | 0.8889 | 0.7778 |
| Macro F1 | 0.8885 | 0.7697 |

## Required final checks

Run:

```powershell
python -m pytest
git status
```

Confirm:

```text
nothing to commit, working tree clean
```

Review the rendered GitHub README and verify that all relative links
open correctly.

Do not rerun:

```text
notebooks/10_final_test_evaluation.ipynb
```

The final test completion marker already documents the one-time
evaluation.
