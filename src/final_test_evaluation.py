from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_DATA_COLUMNS = {
    "chart_group_id",
    "image_path",
    "claim_text",
    "label",
}


def verify_final_evaluation_preconditions(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    selected_metrics: pd.DataFrame,
) -> None:
    """Validate split isolation and the frozen selected configuration."""

    for split_name, split_data in [
        ("train", train_data),
        ("validation", validation_data),
        ("test", test_data),
    ]:
        missing_columns = (
            REQUIRED_DATA_COLUMNS
            - set(split_data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{split_name} split is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if split_data.empty:
            raise ValueError(
                f"{split_name} split cannot be empty."
            )

    train_groups = set(
        train_data["chart_group_id"]
    )

    validation_groups = set(
        validation_data["chart_group_id"]
    )

    test_groups = set(
        test_data["chart_group_id"]
    )

    overlap_checks = {
        "train-validation": (
            train_groups
            & validation_groups
        ),
        "train-test": (
            train_groups
            & test_groups
        ),
        "validation-test": (
            validation_groups
            & test_groups
        ),
    }

    overlapping_pairs = {
        pair_name: sorted(overlap)
        for pair_name, overlap
        in overlap_checks.items()
        if overlap
    }

    if overlapping_pairs:
        raise ValueError(
            "Chart groups overlap across splits: "
            f"{overlapping_pairs}"
        )

    required_metric_columns = {
        "model",
        "config_name",
        "accuracy",
        "macro_f1",
        "evaluation_split",
    }

    missing_metric_columns = (
        required_metric_columns
        - set(selected_metrics.columns)
    )

    if missing_metric_columns:
        raise ValueError(
            "Selected metrics are missing columns: "
            f"{sorted(missing_metric_columns)}"
        )

    if len(selected_metrics) != 1:
        raise ValueError(
            "Selected metrics must contain exactly one row."
        )

    selected_row = selected_metrics.iloc[0]

    if selected_row["config_name"] != "baseline":
        raise ValueError(
            "The frozen final configuration must be baseline."
        )

    if selected_row["evaluation_split"] != "validation":
        raise ValueError(
            "The selected configuration must come from validation."
        )


def assert_test_not_previously_evaluated(
    marker_path: str | Path,
) -> None:
    """Stop accidental repeated final test evaluation."""

    marker = Path(marker_path)

    if marker.exists():
        raise RuntimeError(
            "Final test evaluation was already completed. "
            "Do not rerun it for model selection or tuning. "
            f"Marker: {marker}"
        )


def build_final_predictions_table(
    test_data: pd.DataFrame,
    predicted_labels: Sequence[str],
    predicted_probabilities: np.ndarray,
    label_order: Sequence[str],
) -> pd.DataFrame:
    """Build a transparent row-level final test prediction report."""

    missing_columns = (
        REQUIRED_DATA_COLUMNS
        - set(test_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Test data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    row_count = len(test_data)

    if len(predicted_labels) != row_count:
        raise ValueError(
            "Prediction count must match test rows."
        )

    probabilities = np.asarray(
        predicted_probabilities,
        dtype=float,
    )

    expected_shape = (
        row_count,
        len(label_order),
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "Probability shape must be "
            f"{expected_shape}, got "
            f"{probabilities.shape}."
        )

    predictions = test_data[
        [
            "chart_group_id",
            "image_path",
            "claim_text",
            "label",
        ]
    ].reset_index(drop=True).copy()

    predictions = predictions.rename(
        columns={
            "label": "true_label",
        }
    )

    predictions["predicted_label"] = list(
        predicted_labels
    )

    predictions["confidence"] = (
        probabilities.max(axis=1)
    )

    predictions["correct"] = (
        predictions["true_label"]
        == predictions["predicted_label"]
    )

    for class_index, class_label in enumerate(
        label_order
    ):
        predictions[
            f"probability_{class_label}"
        ] = probabilities[:, class_index]

    predictions["error_type"] = np.where(
        predictions["correct"],
        "correct",
        (
            predictions["true_label"]
            + " -> "
            + predictions["predicted_label"]
        ),
    )

    return predictions


def build_final_evaluation_summary(
    validation_accuracy: float,
    validation_macro_f1: float,
    test_accuracy: float,
    test_macro_f1: float,
) -> pd.DataFrame:
    """Compare frozen validation metrics with the one-time test result."""

    return pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "validation_score": (
                    validation_accuracy
                ),
                "test_score": test_accuracy,
                "test_minus_validation": (
                    test_accuracy
                    - validation_accuracy
                ),
            },
            {
                "metric": "macro_f1",
                "validation_score": (
                    validation_macro_f1
                ),
                "test_score": test_macro_f1,
                "test_minus_validation": (
                    test_macro_f1
                    - validation_macro_f1
                ),
            },
        ]
    )


def write_test_evaluation_marker(
    marker_path: str | Path,
    model_name: str,
    config_name: str,
    test_size: int,
    accuracy: float,
    macro_f1: float,
) -> None:
    """Write the final-evaluation completion marker."""

    marker = Path(marker_path)

    marker.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    marker_payload = {
        "status": "completed",
        "completed_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "model": model_name,
        "config_name": config_name,
        "test_size": int(test_size),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "test_split_policy": (
            "evaluated once after model selection"
        ),
    }

    marker.write_text(
        __import__("json").dumps(
            marker_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
