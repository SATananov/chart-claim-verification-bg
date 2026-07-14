from __future__ import annotations

import csv
import json
from pathlib import Path

from src.final_submission_audit import (
    EXPECTED_CORRECT_COUNT,
    EXPECTED_ERROR_COUNT,
    EXPECTED_TEST_SIZE,
    check_confusion_matrix,
    check_evaluation_summary,
    check_final_metrics,
    check_notebook_sequence,
    check_prediction_reports,
    render_markdown_report,
)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_notebook_sequence_accepts_00_to_10(
    tmp_path: Path,
):
    notebook_directory = (
        tmp_path / "notebooks"
    )

    notebook_directory.mkdir()

    for number in range(11):
        (
            notebook_directory
            / f"{number:02d}_example.ipynb"
        ).write_text(
            "{}",
            encoding="utf-8",
        )

    result = check_notebook_sequence(
        tmp_path
    )

    assert result.status == "PASS"


def test_final_metrics_accept_expected_values(
    tmp_path: Path,
):
    write_csv(
        tmp_path
        / "reports"
        / (
            "multimodal_best_"
            "validation_metrics.csv"
        ),
        [
            "model",
            "config_name",
            "accuracy",
            "macro_f1",
            "evaluation_split",
        ],
        [
            {
                "model": "selected",
                "config_name": "baseline",
                "accuracy": (
                    0.8888888888888888
                ),
                "macro_f1": (
                    0.8885003885003885
                ),
                "evaluation_split": (
                    "validation"
                ),
            }
        ],
    )

    write_csv(
        tmp_path
        / "reports"
        / "final_test_metrics.csv",
        [
            "model",
            "config_name",
            "accuracy",
            "macro_f1",
            "evaluation_split",
            "test_size",
        ],
        [
            {
                "model": "final",
                "config_name": "baseline",
                "accuracy": (
                    0.7777777777777778
                ),
                "macro_f1": (
                    0.7696969696969697
                ),
                "evaluation_split": (
                    "test"
                ),
                "test_size": 54,
            }
        ],
    )

    results = check_final_metrics(
        tmp_path
    )

    assert [
        result.status
        for result in results
    ] == ["PASS", "PASS"]


def test_evaluation_summary_checks_gap(
    tmp_path: Path,
):
    write_csv(
        tmp_path
        / "reports"
        / "final_evaluation_summary.csv",
        [
            "metric",
            "validation_score",
            "test_score",
            "test_minus_validation",
        ],
        [
            {
                "metric": "accuracy",
                "validation_score": (
                    0.8888888888888888
                ),
                "test_score": (
                    0.7777777777777778
                ),
                "test_minus_validation": (
                    -0.11111111111111105
                ),
            },
            {
                "metric": "macro_f1",
                "validation_score": (
                    0.8885003885003885
                ),
                "test_score": (
                    0.7696969696969697
                ),
                "test_minus_validation": (
                    -0.11880341880341883
                ),
            },
        ],
    )

    result = check_evaluation_summary(
        tmp_path
    )

    assert result.status == "PASS"


def test_confusion_matrix_checks_total_and_diagonal(
    tmp_path: Path,
):
    write_csv(
        tmp_path
        / "reports"
        / "final_test_confusion_matrix.csv",
        [
            "",
            "predicted_supported",
            "predicted_refuted",
            (
                "predicted_"
                "not_enough_information"
            ),
        ],
        [
            {
                "": "actual_supported",
                "predicted_supported": 15,
                "predicted_refuted": 0,
                (
                    "predicted_"
                    "not_enough_information"
                ): 3,
            },
            {
                "": "actual_refuted",
                "predicted_supported": 0,
                "predicted_refuted": 18,
                (
                    "predicted_"
                    "not_enough_information"
                ): 0,
            },
            {
                "": (
                    "actual_"
                    "not_enough_information"
                ),
                "predicted_supported": 0,
                "predicted_refuted": 9,
                (
                    "predicted_"
                    "not_enough_information"
                ): 9,
            },
        ],
    )

    result = check_confusion_matrix(
        tmp_path
    )

    assert result.status == "PASS"


def test_prediction_reports_check_counts(
    tmp_path: Path,
):
    prediction_rows = []

    for index in range(
        EXPECTED_TEST_SIZE
    ):
        is_correct = (
            index
            < EXPECTED_CORRECT_COUNT
        )

        prediction_rows.append(
            {
                "correct": str(
                    is_correct
                ),
                "confidence": 0.8,
                "probability_supported": 0.8,
                "probability_refuted": 0.1,
                (
                    "probability_"
                    "not_enough_information"
                ): 0.1,
            }
        )

    write_csv(
        tmp_path
        / "reports"
        / "final_test_predictions.csv",
        [
            "correct",
            "confidence",
            "probability_supported",
            "probability_refuted",
            (
                "probability_"
                "not_enough_information"
            ),
        ],
        prediction_rows,
    )

    write_csv(
        tmp_path
        / "reports"
        / "final_test_errors.csv",
        ["error_type"],
        [
            {
                "error_type": "example"
            }
            for _ in range(
                EXPECTED_ERROR_COUNT
            )
        ],
    )

    result = check_prediction_reports(
        tmp_path
    )

    assert result.status == "PASS"


def test_markdown_report_states_no_test_rerun():
    result = {
        "status": "PASS",
        "passed_checks": 1,
        "failed_checks": 0,
        "policy": "Stored artifacts only.",
        "checks": [
            {
                "name": "example",
                "status": "PASS",
                "details": "Example passed.",
            }
        ],
    }

    report = render_markdown_report(
        result
    )

    assert (
        "does not load the trained model"
        in report
    )
    assert (
        "was not executed by this audit"
        in report
    )
