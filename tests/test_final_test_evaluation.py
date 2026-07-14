import json

import numpy as np
import pandas as pd
import pytest

from src.final_test_evaluation import (
    assert_test_not_previously_evaluated,
    build_final_evaluation_summary,
    build_final_predictions_table,
    verify_final_evaluation_preconditions,
    write_test_evaluation_marker,
)


def make_split(
    group_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chart_group_id": group_name,
                "image_path": (
                    f"{group_name}.png"
                ),
                "claim_text": "Example claim",
                "label": "supported",
            }
        ]
    )


def selected_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": (
                    "Tuned Keras multimodal "
                    "neural network"
                ),
                "config_name": "baseline",
                "accuracy": 0.88,
                "macro_f1": 0.87,
                "evaluation_split": (
                    "validation"
                ),
            }
        ]
    )


def test_final_preconditions_accept_clean_splits():
    verify_final_evaluation_preconditions(
        train_data=make_split("train-1"),
        validation_data=make_split("validation-1"),
        test_data=make_split("test-1"),
        selected_metrics=selected_metrics(),
    )


def test_final_preconditions_reject_group_overlap():
    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        verify_final_evaluation_preconditions(
            train_data=make_split("shared"),
            validation_data=make_split(
                "validation-1"
            ),
            test_data=make_split("shared"),
            selected_metrics=(
                selected_metrics()
            ),
        )


def test_final_prediction_table_marks_error():
    test_data = pd.DataFrame(
        [
            {
                "chart_group_id": "test-1",
                "image_path": "chart.png",
                "claim_text": "Example claim",
                "label": "refuted",
            }
        ]
    )

    table = build_final_predictions_table(
        test_data=test_data,
        predicted_labels=["supported"],
        predicted_probabilities=np.asarray(
            [[0.70, 0.20, 0.10]]
        ),
        label_order=[
            "supported",
            "refuted",
            "not_enough_information",
        ],
    )

    assert table.loc[0, "correct"] == False
    assert table.loc[
        0,
        "error_type",
    ] == "refuted -> supported"
    assert np.isclose(
        table.loc[0, "confidence"],
        0.70,
    )


def test_final_summary_calculates_gap():
    summary = build_final_evaluation_summary(
        validation_accuracy=0.90,
        validation_macro_f1=0.88,
        test_accuracy=0.80,
        test_macro_f1=0.78,
    )

    accuracy_gap = summary.loc[
        summary["metric"] == "accuracy",
        "test_minus_validation",
    ].iloc[0]

    assert np.isclose(
        accuracy_gap,
        -0.10,
    )


def test_final_marker_blocks_repeat(
    tmp_path,
):
    marker_path = (
        tmp_path
        / "final_test_evaluation_completed.json"
    )

    assert_test_not_previously_evaluated(
        marker_path
    )

    write_test_evaluation_marker(
        marker_path=marker_path,
        model_name="final model",
        config_name="baseline",
        test_size=54,
        accuracy=0.80,
        macro_f1=0.79,
    )

    marker_payload = json.loads(
        marker_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        marker_payload["status"]
        == "completed"
    )

    with pytest.raises(
        RuntimeError,
        match="already completed",
    ):
        assert_test_not_previously_evaluated(
            marker_path
        )
