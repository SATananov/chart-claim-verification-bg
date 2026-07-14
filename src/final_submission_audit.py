from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_AUDIT_PATHS = {
    "docs/final_submission_audit.md",
    "scripts/run_final_submission_audit.py",
    "src/final_submission_audit.py",
    "tests/test_final_submission_audit.py",
    "reports/final_submission_audit.json",
    "reports/final_submission_audit.md",
    "notebooks/09_evaluation_and_error_analysis.ipynb",
    "notebooks/10_final_results.ipynb",
}

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "docs/final_project_report.md",
    "docs/submission_checklist.md",
    "reports/model_validation_comparison.csv",
    "reports/multimodal_best_validation_metrics.csv",
    "reports/final_evaluation_summary.csv",
    "reports/final_test_metrics.csv",
    "reports/final_test_confusion_matrix.csv",
    "reports/final_test_predictions.csv",
    "reports/final_test_errors.csv",
    "reports/final_test_evaluation_completed.json",
    "src/final_test_evaluation.py",
    "src/multimodal_neural_network.py",
    "src/multimodal_tuning.py",
    "tests/test_final_test_evaluation.py",
    "notebooks/10_final_test_evaluation.ipynb",
]

EXPECTED_VALIDATION_ACCURACY = 0.8888888888888888
EXPECTED_VALIDATION_MACRO_F1 = 0.8885003885003885
EXPECTED_TEST_ACCURACY = 0.7777777777777778
EXPECTED_TEST_MACRO_F1 = 0.7696969696969697
EXPECTED_TEST_SIZE = 54
EXPECTED_CORRECT_COUNT = 42
EXPECTED_ERROR_COUNT = 12


@dataclass(frozen=True)
class AuditCheck:
    """One final-submission audit result."""

    name: str
    status: str
    details: str


def _check(
    name: str,
    passed: bool,
    success_details: str,
    failure_details: str,
) -> AuditCheck:
    return AuditCheck(
        name=name,
        status="PASS" if passed else "FAIL",
        details=(
            success_details
            if passed
            else failure_details
        ),
    )


def _read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _close(
    first: float,
    second: float,
    tolerance: float = 1e-9,
) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def check_required_files(
    project_root: Path,
) -> AuditCheck:
    """Verify that the final submission evidence exists."""

    missing = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (
            project_root
            / relative_path
        ).is_file()
    ]

    return _check(
        name="required_files",
        passed=not missing,
        success_details=(
            f"All {len(REQUIRED_FILES)} "
            "required files are present."
        ),
        failure_details=(
            "Missing required files: "
            + ", ".join(missing)
        ),
    )


def check_notebook_sequence(
    project_root: Path,
) -> AuditCheck:
    """Verify one notebook prefix for every step from 00 to 10."""

    notebook_directory = (
        project_root / "notebooks"
    )

    found_prefixes = []

    if notebook_directory.is_dir():
        for notebook_path in (
            notebook_directory.glob("*.ipynb")
        ):
            prefix = notebook_path.name[:2]

            if prefix.isdigit():
                found_prefixes.append(
                    int(prefix)
                )

    expected = list(range(11))

    missing = [
        number
        for number in expected
        if found_prefixes.count(number) == 0
    ]

    duplicates = [
        number
        for number in expected
        if found_prefixes.count(number) > 1
    ]

    passed = (
        not missing
        and not duplicates
    )

    return _check(
        name="notebook_sequence",
        passed=passed,
        success_details=(
            "Notebook prefixes 00 through 10 "
            "are present exactly once."
        ),
        failure_details=(
            f"Missing prefixes: {missing}; "
            f"duplicate prefixes: {duplicates}."
        ),
    )


def check_final_metrics(
    project_root: Path,
) -> list[AuditCheck]:
    """Check frozen validation and final test metrics."""

    validation_rows = _read_csv_rows(
        project_root
        / "reports"
        / "multimodal_best_validation_metrics.csv"
    )

    test_rows = _read_csv_rows(
        project_root
        / "reports"
        / "final_test_metrics.csv"
    )

    checks = []

    validation_ok = (
        len(validation_rows) == 1
        and validation_rows[0]["config_name"]
        == "baseline"
        and validation_rows[0][
            "evaluation_split"
        ]
        == "validation"
        and _close(
            float(
                validation_rows[0][
                    "accuracy"
                ]
            ),
            EXPECTED_VALIDATION_ACCURACY,
        )
        and _close(
            float(
                validation_rows[0][
                    "macro_f1"
                ]
            ),
            EXPECTED_VALIDATION_MACRO_F1,
        )
    )

    checks.append(
        _check(
            name="frozen_validation_metrics",
            passed=validation_ok,
            success_details=(
                "Frozen validation metrics match "
                "the selected baseline model."
            ),
            failure_details=(
                "Frozen validation metrics or "
                "configuration do not match."
            ),
        )
    )

    test_ok = (
        len(test_rows) == 1
        and test_rows[0]["config_name"]
        == "baseline"
        and test_rows[0][
            "evaluation_split"
        ]
        == "test"
        and int(
            test_rows[0]["test_size"]
        )
        == EXPECTED_TEST_SIZE
        and _close(
            float(
                test_rows[0]["accuracy"]
            ),
            EXPECTED_TEST_ACCURACY,
        )
        and _close(
            float(
                test_rows[0]["macro_f1"]
            ),
            EXPECTED_TEST_MACRO_F1,
        )
    )

    checks.append(
        _check(
            name="final_test_metrics",
            passed=test_ok,
            success_details=(
                "Final test metrics match the "
                "documented one-time evaluation."
            ),
            failure_details=(
                "Final test metrics do not match "
                "the documented result."
            ),
        )
    )

    return checks


def check_evaluation_summary(
    project_root: Path,
) -> AuditCheck:
    """Verify the reported validation-to-test gaps."""

    rows = _read_csv_rows(
        project_root
        / "reports"
        / "final_evaluation_summary.csv"
    )

    by_metric = {
        row["metric"]: row
        for row in rows
    }

    expected = {
        "accuracy": (
            EXPECTED_VALIDATION_ACCURACY,
            EXPECTED_TEST_ACCURACY,
        ),
        "macro_f1": (
            EXPECTED_VALIDATION_MACRO_F1,
            EXPECTED_TEST_MACRO_F1,
        ),
    }

    passed = set(by_metric) == set(expected)

    if passed:
        for metric_name, (
            validation_score,
            test_score,
        ) in expected.items():
            row = by_metric[metric_name]

            passed = passed and all(
                [
                    _close(
                        float(
                            row[
                                "validation_score"
                            ]
                        ),
                        validation_score,
                    ),
                    _close(
                        float(
                            row["test_score"]
                        ),
                        test_score,
                    ),
                    _close(
                        float(
                            row[
                                "test_minus_validation"
                            ]
                        ),
                        (
                            test_score
                            - validation_score
                        ),
                    ),
                ]
            )

    return _check(
        name="generalization_gap",
        passed=passed,
        success_details=(
            "Validation-to-test gaps are "
            "arithmetically consistent."
        ),
        failure_details=(
            "The final evaluation summary is "
            "inconsistent with stored metrics."
        ),
    )


def check_confusion_matrix(
    project_root: Path,
) -> AuditCheck:
    """Verify total and diagonal counts in the final matrix."""

    rows = _read_csv_rows(
        project_root
        / "reports"
        / "final_test_confusion_matrix.csv"
    )

    prediction_columns = [
        "predicted_supported",
        "predicted_refuted",
        "predicted_not_enough_information",
    ]

    total = 0
    diagonal = 0

    expected_diagonal_column = {
        "actual_supported": (
            "predicted_supported"
        ),
        "actual_refuted": (
            "predicted_refuted"
        ),
        "actual_not_enough_information": (
            "predicted_not_enough_information"
        ),
    }

    passed = len(rows) == 3

    if passed:
        for row in rows:
            actual_label = row.get("", "")

            if (
                actual_label
                not in expected_diagonal_column
            ):
                passed = False
                break

            for column in prediction_columns:
                total += int(row[column])

            diagonal += int(
                row[
                    expected_diagonal_column[
                        actual_label
                    ]
                ]
            )

    passed = (
        passed
        and total == EXPECTED_TEST_SIZE
        and diagonal == EXPECTED_CORRECT_COUNT
    )

    return _check(
        name="final_confusion_matrix",
        passed=passed,
        success_details=(
            "Confusion matrix totals 54 examples "
            "with 42 correct predictions."
        ),
        failure_details=(
            f"Confusion matrix total={total}, "
            f"diagonal={diagonal}."
        ),
    )


def check_prediction_reports(
    project_root: Path,
) -> AuditCheck:
    """Verify row-level test prediction integrity."""

    prediction_rows = _read_csv_rows(
        project_root
        / "reports"
        / "final_test_predictions.csv"
    )

    error_rows = _read_csv_rows(
        project_root
        / "reports"
        / "final_test_errors.csv"
    )

    correct_count = 0
    probability_ok = True

    probability_columns = [
        "probability_supported",
        "probability_refuted",
        (
            "probability_"
            "not_enough_information"
        ),
    ]

    for row in prediction_rows:
        is_correct = (
            row["correct"].strip().lower()
            == "true"
        )

        correct_count += int(is_correct)

        confidence = float(
            row["confidence"]
        )

        probabilities = [
            float(row[column])
            for column in probability_columns
        ]

        probability_ok = (
            probability_ok
            and 0.0 <= confidence <= 1.0
            and all(
                0.0 <= value <= 1.0
                for value in probabilities
            )
            and _close(
                sum(probabilities),
                1.0,
                tolerance=1e-5,
            )
        )

    passed = all(
        [
            len(prediction_rows)
            == EXPECTED_TEST_SIZE,
            correct_count
            == EXPECTED_CORRECT_COUNT,
            len(error_rows)
            == EXPECTED_ERROR_COUNT,
            probability_ok,
        ]
    )

    return _check(
        name="row_level_test_reports",
        passed=passed,
        success_details=(
            "Prediction and error reports contain "
            "54 rows, 42 correct and 12 errors."
        ),
        failure_details=(
            "Row counts, correctness counts or "
            "probabilities are inconsistent."
        ),
    )


def check_completion_marker(
    project_root: Path,
) -> AuditCheck:
    """Verify the one-time final-test completion marker."""

    marker = _load_json(
        project_root
        / "reports"
        / "final_test_evaluation_completed.json"
    )

    passed = all(
        [
            marker.get("status")
            == "completed",
            marker.get("config_name")
            == "baseline",
            int(
                marker.get(
                    "test_size",
                    -1,
                )
            )
            == EXPECTED_TEST_SIZE,
            _close(
                float(
                    marker.get(
                        "accuracy",
                        -1.0,
                    )
                ),
                EXPECTED_TEST_ACCURACY,
            ),
            _close(
                float(
                    marker.get(
                        "macro_f1",
                        -1.0,
                    )
                ),
                EXPECTED_TEST_MACRO_F1,
            ),
        ]
    )

    return _check(
        name="one_time_test_marker",
        passed=passed,
        success_details=(
            "Completion marker confirms the "
            "frozen one-time test evaluation."
        ),
        failure_details=(
            "Completion marker is missing or "
            "inconsistent with final metrics."
        ),
    )


def check_documentation(
    project_root: Path,
) -> AuditCheck:
    """Verify final metrics and research references in documentation."""

    readme = (
        project_root / "README.md"
    ).read_text(
        encoding="utf-8"
    )

    report = (
        project_root
        / "docs"
        / "final_project_report.md"
    ).read_text(
        encoding="utf-8"
    )

    required_strings = [
        "0.8889",
        "0.8885",
        "0.7778",
        "0.7697",
        "FEVER",
        "DVQA",
        "PlotQA",
        "ChartQA",
        "DePlot",
        "not_enough_information",
    ]

    missing = [
        value
        for value in required_strings
        if (
            value not in readme
            or value not in report
        )
    ]

    return _check(
        name="documentation_consistency",
        passed=not missing,
        success_details=(
            "README and final report contain the "
            "final metrics and research comparison."
        ),
        failure_details=(
            "Missing documentation values: "
            + ", ".join(missing)
        ),
    )


def check_final_notebook_guard(
    project_root: Path,
) -> AuditCheck:
    """Inspect notebook source without executing it."""

    notebook_path = (
        project_root
        / "notebooks"
        / "10_final_test_evaluation.ipynb"
    )

    notebook = _load_json(
        notebook_path
    )

    code_source = "\n".join(
        "".join(
            cell.get("source", [])
        )
        for cell in notebook.get(
            "cells",
            [],
        )
        if cell.get("cell_type") == "code"
    )

    guard_position = code_source.find(
        "assert_test_not_previously_evaluated"
    )

    prediction_position = code_source.find(
        "final_model.predict"
    )

    has_error_output = any(
        output.get("output_type")
        == "error"
        for cell in notebook.get(
            "cells",
            [],
        )
        for output in cell.get(
            "outputs",
            [],
        )
    )

    passed = all(
        [
            guard_position >= 0,
            prediction_position >= 0,
            guard_position < prediction_position,
            not has_error_output,
        ]
    )

    return _check(
        name="final_notebook_guard",
        passed=passed,
        success_details=(
            "Final notebook contains the marker "
            "guard before prediction and no error output."
        ),
        failure_details=(
            "Final notebook guard order or stored "
            "outputs are not valid."
        ),
    )


def _run_git(
    project_root: Path,
    arguments: Iterable[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _porcelain_paths(
    output: str,
) -> set[str]:
    paths = set()

    for line in output.splitlines():
        if not line:
            continue

        path_text = line[3:].strip()

        if " -> " in path_text:
            path_text = path_text.split(
                " -> ",
                maxsplit=1,
            )[1]

        paths.add(
            path_text.strip('"')
        )

    return paths


def check_git_state(
    project_root: Path,
) -> list[AuditCheck]:
    """Check branch, upstream state and unrelated local changes."""

    inside = _run_git(
        project_root,
        [
            "rev-parse",
            "--is-inside-work-tree",
        ],
    )

    repository_ok = (
        inside.returncode == 0
        and inside.stdout.strip()
        == "true"
    )

    checks = [
        _check(
            name="git_repository",
            passed=repository_ok,
            success_details=(
                "Project is inside a Git repository."
            ),
            failure_details=(
                "Git repository check failed."
            ),
        )
    ]

    if not repository_ok:
        return checks

    branch = _run_git(
        project_root,
        [
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ],
    ).stdout.strip()

    checks.append(
        _check(
            name="git_branch",
            passed=branch == "main",
            success_details=(
                "Current branch is main."
            ),
            failure_details=(
                f"Current branch is {branch!r}, "
                "expected 'main'."
            ),
        )
    )

    upstream = _run_git(
        project_root,
        [
            "rev-list",
            "--left-right",
            "--count",
            "@{upstream}...HEAD",
        ],
    )

    upstream_ok = False
    upstream_details = (
        "No usable upstream comparison."
    )

    if upstream.returncode == 0:
        parts = (
            upstream.stdout
            .strip()
            .split()
        )

        if len(parts) == 2:
            behind = int(parts[0])
            ahead = int(parts[1])
            upstream_ok = (
                behind == 0
                and ahead == 0
            )
            upstream_details = (
                f"behind={behind}, ahead={ahead}"
            )

    checks.append(
        _check(
            name="git_upstream_sync",
            passed=upstream_ok,
            success_details=(
                "main is synchronized with its "
                f"upstream ({upstream_details})."
            ),
            failure_details=(
                "Branch is not synchronized with "
                f"upstream ({upstream_details})."
            ),
        )
    )

    status = _run_git(
        project_root,
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
    )

    changed_paths = _porcelain_paths(
        status.stdout
    )

    unrelated_paths = sorted(
        changed_paths
        - EXPECTED_AUDIT_PATHS
    )

    checks.append(
        _check(
            name="unrelated_local_changes",
            passed=not unrelated_paths,
            success_details=(
                "No unrelated local changes were "
                "found outside Step 014 files."
            ),
            failure_details=(
                "Unrelated local changes: "
                + ", ".join(unrelated_paths)
            ),
        )
    )

    tracked_files = _run_git(
        project_root,
        [
            "ls-files",
        ],
    )

    tracked_paths = {
        line.strip()
        for line in (
            tracked_files.stdout
            .splitlines()
        )
        if line.strip()
    }

    forbidden_tracked = sorted(
        path
        for path in tracked_paths
        if (
            path.endswith(".keras")
            or (
                path.startswith(
                    "data/generated/"
                )
                and path.lower().endswith(
                    ".png"
                )
            )
        )
    )

    checks.append(
        _check(
            name="ignored_generated_artifacts",
            passed=not forbidden_tracked,
            success_details=(
                "No trained Keras model or generated "
                "chart PNG is tracked by Git."
            ),
            failure_details=(
                "Generated artifacts are tracked: "
                + ", ".join(
                    forbidden_tracked
                )
            ),
        )
    )

    return checks


def run_final_submission_audit(
    project_root: Path,
) -> dict[str, Any]:
    """Run a read-only audit over stored project artifacts."""

    checks: list[AuditCheck] = []

    required_check = check_required_files(
        project_root
    )
    checks.append(required_check)

    if required_check.status == "PASS":
        checks.extend(
            [
                check_notebook_sequence(
                    project_root
                ),
                *check_final_metrics(
                    project_root
                ),
                check_evaluation_summary(
                    project_root
                ),
                check_confusion_matrix(
                    project_root
                ),
                check_prediction_reports(
                    project_root
                ),
                check_completion_marker(
                    project_root
                ),
                check_documentation(
                    project_root
                ),
                check_final_notebook_guard(
                    project_root
                ),
            ]
        )

    checks.extend(
        check_git_state(
            project_root
        )
    )

    failed_checks = [
        check.name
        for check in checks
        if check.status == "FAIL"
    ]

    return {
        "audit_name": (
            "Step 014 Final Submission Audit"
        ),
        "project": (
            "chart-claim-verification-bg"
        ),
        "policy": (
            "Stored artifacts only; the final "
            "test model is not loaded or evaluated."
        ),
        "status": (
            "PASS"
            if not failed_checks
            else "FAIL"
        ),
        "passed_checks": sum(
            check.status == "PASS"
            for check in checks
        ),
        "failed_checks": len(
            failed_checks
        ),
        "failed_check_names": (
            failed_checks
        ),
        "checks": [
            asdict(check)
            for check in checks
        ],
    }


def render_markdown_report(
    audit_result: dict[str, Any],
) -> str:
    """Render the JSON-compatible audit as Markdown."""

    lines = [
        "# Final Submission Audit",
        "",
        (
            f"**Status:** "
            f"{audit_result['status']}"
        ),
        "",
        (
            f"**Passed checks:** "
            f"{audit_result['passed_checks']}"
        ),
        "",
        (
            f"**Failed checks:** "
            f"{audit_result['failed_checks']}"
        ),
        "",
        "## Policy",
        "",
        audit_result["policy"],
        "",
        "This audit reads stored reports, documentation, "
        "notebook JSON and Git metadata. It does not load "
        "the trained model, rebuild predictions or open the "
        "test split for another evaluation.",
        "",
        "## Checks",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]

    for check in audit_result["checks"]:
        details = (
            str(check["details"])
            .replace("|", "\\|")
            .replace("\n", " ")
        )

        lines.append(
            "| "
            + str(check["name"])
            + " | "
            + str(check["status"])
            + " | "
            + details
            + " |"
        )

    lines.extend(
        [
            "",
            "## Final stored result",
            "",
            "- Validation accuracy: `0.8889`",
            "- Validation macro F1: `0.8885`",
            "- Final test accuracy: `0.7778`",
            "- Final test macro F1: `0.7697`",
            "- Final test examples: `54`",
            "- Correct final predictions: `42`",
            "- Final test errors: `12`",
            "",
            "The final test notebook was not executed by "
            "this audit.",
            "",
        ]
    )

    return "\n".join(lines)


def write_audit_reports(
    project_root: Path,
    audit_result: dict[str, Any],
) -> tuple[Path, Path]:
    """Write JSON and Markdown audit evidence."""

    reports_directory = (
        project_root / "reports"
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        reports_directory
        / "final_submission_audit.json"
    )

    markdown_path = (
        reports_directory
        / "final_submission_audit.md"
    )

    json_path.write_text(
        json.dumps(
            audit_result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        render_markdown_report(
            audit_result
        ),
        encoding="utf-8",
    )

    return json_path, markdown_path
