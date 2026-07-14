# Step 014 — Final Submission Audit

This step performs a read-only audit of the completed repository.

It does **not**:

- load the trained multimodal model;
- call `model.predict`;
- rebuild the final test predictions;
- rerun `notebooks/10_final_test_evaluation.ipynb`;
- change the frozen model or its hyperparameters.

## What the audit checks

The audit verifies:

1. required final files and reports;
2. notebook numbering from `00` through `10`;
3. frozen validation metrics;
4. final test metrics;
5. validation-to-test gap arithmetic;
6. final confusion-matrix totals;
7. row-level prediction and error counts;
8. probability consistency;
9. the one-time final-test marker;
10. README and final-report consistency;
11. the final notebook guard order without executing it;
12. Git branch and upstream synchronization;
13. absence of unrelated local changes;
14. absence of tracked `.keras` models and generated chart PNG files.

## Expected local changes

Before the Step 014 commit, the audit allows only these paths to be
changed or untracked:

```text
docs/final_submission_audit.md
scripts/run_final_submission_audit.py
src/final_submission_audit.py
tests/test_final_submission_audit.py
reports/final_submission_audit.json
reports/final_submission_audit.md
```

Any unrelated local modification causes the audit to fail.

## Run

```powershell
python scripts/run_final_submission_audit.py
```

A successful result prints:

```json
{
  "status": "PASS",
  "failed_checks": 0
}
```

The command writes:

```text
reports/final_submission_audit.json
reports/final_submission_audit.md
```

Then run:

```powershell
python -m pytest
git status --short
```

The final test notebook must not be executed again.
