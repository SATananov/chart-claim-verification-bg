# Final Submission Audit

**Status:** PASS

**Passed checks:** 15

**Failed checks:** 0

## Policy

Stored artifacts only; the final test model is not loaded or evaluated.

This audit reads stored reports, documentation, notebook JSON and Git metadata. It does not load the trained model, rebuild predictions or open the test split for another evaluation.

## Checks

| Check | Status | Details |
|---|---|---|
| required_files | PASS | All 18 required files are present. |
| notebook_sequence | PASS | Notebook prefixes 00 through 10 are present exactly once. |
| frozen_validation_metrics | PASS | Frozen validation metrics match the selected baseline model. |
| final_test_metrics | PASS | Final test metrics match the documented one-time evaluation. |
| generalization_gap | PASS | Validation-to-test gaps are arithmetically consistent. |
| final_confusion_matrix | PASS | Confusion matrix totals 54 examples with 42 correct predictions. |
| row_level_test_reports | PASS | Prediction and error reports contain 54 rows, 42 correct and 12 errors. |
| one_time_test_marker | PASS | Completion marker confirms the frozen one-time test evaluation. |
| documentation_consistency | PASS | README and final report contain the final metrics and research comparison. |
| final_notebook_guard | PASS | Final notebook contains the marker guard before prediction and no error output. |
| git_repository | PASS | Project is inside a Git repository. |
| git_branch | PASS | Current branch is main. |
| git_upstream_sync | PASS | main is synchronized with its upstream (behind=0, ahead=0). |
| unrelated_local_changes | PASS | No unrelated local changes were found outside Step 014 files. |
| ignored_generated_artifacts | PASS | No trained Keras model or generated chart PNG is tracked by Git. |

## Final stored result

- Validation accuracy: `0.8889`
- Validation macro F1: `0.8885`
- Final test accuracy: `0.7778`
- Final test macro F1: `0.7697`
- Final test examples: `54`
- Correct final predictions: `42`
- Final test errors: `12`

The final test notebook was not executed by this audit.
