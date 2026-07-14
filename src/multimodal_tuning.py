from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def get_tuning_configurations() -> list[dict[str, object]]:
    """Return a small, controlled multimodal search space."""

    return [
        {
            "config_name": "baseline",
            "embedding_dimension": 32,
            "text_units": 32,
            "combined_units": 64,
            "dropout_rate": 0.30,
            "learning_rate": 0.001,
        },
        {
            "config_name": "lower_dropout",
            "embedding_dimension": 32,
            "text_units": 32,
            "combined_units": 64,
            "dropout_rate": 0.15,
            "learning_rate": 0.001,
        },
        {
            "config_name": "wider_fusion",
            "embedding_dimension": 48,
            "text_units": 48,
            "combined_units": 96,
            "dropout_rate": 0.30,
            "learning_rate": 0.0007,
        },
    ]


def select_best_configuration(
    tuning_results: pd.DataFrame,
) -> pd.Series:
    """Select by macro F1, then accuracy, then shorter training."""

    required_columns = {
        "config_name",
        "accuracy",
        "macro_f1",
        "epochs_completed",
    }

    missing_columns = (
        required_columns
        - set(tuning_results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing tuning result columns: "
            f"{sorted(missing_columns)}"
        )

    if tuning_results.empty:
        raise ValueError(
            "Tuning results cannot be empty."
        )

    ranked_results = tuning_results.sort_values(
        by=[
            "macro_f1",
            "accuracy",
            "epochs_completed",
            "config_name",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        kind="stable",
    )

    return ranked_results.iloc[0].copy()


def build_error_analysis_table(
    validation_data: pd.DataFrame,
    predicted_labels: Sequence[str],
    predicted_probabilities: np.ndarray,
    label_order: Sequence[str],
) -> pd.DataFrame:
    """Create one transparent prediction row per validation example."""

    required_columns = {
        "chart_group_id",
        "image_path",
        "claim_text",
        "label",
    }

    missing_columns = (
        required_columns
        - set(validation_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing validation columns: "
            f"{sorted(missing_columns)}"
        )

    row_count = len(validation_data)

    if len(predicted_labels) != row_count:
        raise ValueError(
            "Prediction count must match validation rows."
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

    analysis = validation_data[
        [
            "chart_group_id",
            "image_path",
            "claim_text",
            "label",
        ]
    ].reset_index(drop=True).copy()

    analysis = analysis.rename(
        columns={
            "label": "true_label",
        }
    )

    analysis["predicted_label"] = list(
        predicted_labels
    )

    analysis["confidence"] = (
        probabilities.max(axis=1)
    )

    analysis["correct"] = (
        analysis["true_label"]
        == analysis["predicted_label"]
    )

    for class_index, class_label in enumerate(
        label_order
    ):
        analysis[
            f"probability_{class_label}"
        ] = probabilities[:, class_index]

    analysis["error_type"] = np.where(
        analysis["correct"],
        "correct",
        (
            analysis["true_label"]
            + " -> "
            + analysis["predicted_label"]
        ),
    )

    return analysis


def summarize_validation_errors(
    analysis_table: pd.DataFrame,
) -> pd.DataFrame:
    """Count each true-label to predicted-label error pair."""

    required_columns = {
        "true_label",
        "predicted_label",
        "correct",
    }

    missing_columns = (
        required_columns
        - set(analysis_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing analysis columns: "
            f"{sorted(missing_columns)}"
        )

    errors = analysis_table.loc[
        ~analysis_table["correct"]
    ]

    if errors.empty:
        return pd.DataFrame(
            columns=[
                "true_label",
                "predicted_label",
                "error_count",
            ]
        )

    summary = (
        errors.groupby(
            [
                "true_label",
                "predicted_label",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="error_count"
        )
        .sort_values(
            by=[
                "error_count",
                "true_label",
                "predicted_label",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return summary
