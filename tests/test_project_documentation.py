from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (
        PROJECT_ROOT
        / relative_path
    ).read_text(
        encoding="utf-8"
    )


def test_readme_contains_final_project_sections():
    readme = read_text("README.md")

    required_sections = [
        "# Chart Claim Verification",
        "## Final result",
        "## Research question",
        "## Data",
        "## Model comparison",
        "## Previous research",
        "## Scientific integrity",
        "## Limitations",
        "## Conclusion",
    ]

    for section in required_sections:
        assert section in readme


def test_final_report_contains_research_and_results():
    report = read_text(
        "docs/final_project_report.md"
    )

    required_content = [
        "## 4. Previous research",
        "FEVER",
        "DVQA",
        "PlotQA",
        "ChartQA",
        "DePlot",
        "## 10. Final one-time test evaluation",
        "## 14. Limitations",
        "## 16. Conclusion",
    ]

    for item in required_content:
        assert item in report


def test_documented_final_metrics_match_report_csv():
    metrics_path = (
        PROJECT_ROOT
        / "reports"
        / "final_test_metrics.csv"
    )

    with metrics_path.open(
        encoding="utf-8",
        newline="",
    ) as metrics_file:
        rows = list(
            csv.DictReader(metrics_file)
        )

    assert len(rows) == 1

    accuracy = float(
        rows[0]["accuracy"]
    )

    macro_f1 = float(
        rows[0]["macro_f1"]
    )

    expected_accuracy = f"{accuracy:.4f}"
    expected_macro_f1 = f"{macro_f1:.4f}"

    readme = read_text("README.md")
    report = read_text(
        "docs/final_project_report.md"
    )

    assert expected_accuracy in readme
    assert expected_macro_f1 in readme
    assert expected_accuracy in report
    assert expected_macro_f1 in report


def test_final_evaluation_marker_is_completed():
    marker_path = (
        PROJECT_ROOT
        / "reports"
        / "final_test_evaluation_completed.json"
    )

    marker = json.loads(
        marker_path.read_text(
            encoding="utf-8"
        )
    )

    assert marker["status"] == "completed"
    assert marker["config_name"] == "baseline"
    assert marker["test_size"] == 54


def test_submission_checklist_exists():
    checklist = read_text(
        "docs/submission_checklist.md"
    )

    assert "## Assessment evidence" in checklist
    assert "38 passed" in checklist
    assert (
        "notebooks/10_final_test_evaluation.ipynb"
        in checklist
    )
