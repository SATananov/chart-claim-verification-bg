import numpy as np
import pandas as pd

from src.multimodal_tuning import (
    build_error_analysis_table,
    get_tuning_configurations,
    select_best_configuration,
    summarize_validation_errors,
)


def test_tuning_configurations_are_small_and_unique():
    configurations = get_tuning_configurations()

    names = [
        configuration["config_name"]
        for configuration in configurations
    ]

    assert len(configurations) == 3
    assert len(names) == len(set(names))
    assert "baseline" in names

    for configuration in configurations:
        assert (
            configuration[
                "embedding_dimension"
            ]
            > 0
        )
        assert (
            0
            < configuration["dropout_rate"]
            < 1
        )
        assert (
            configuration["learning_rate"]
            > 0
        )


def test_best_configuration_uses_macro_f1_first():
    results = pd.DataFrame(
        [
            {
                "config_name": "first",
                "accuracy": 0.90,
                "macro_f1": 0.80,
                "epochs_completed": 8,
            },
            {
                "config_name": "second",
                "accuracy": 0.85,
                "macro_f1": 0.84,
                "epochs_completed": 12,
            },
        ]
    )

    best = select_best_configuration(
        results
    )

    assert best["config_name"] == "second"


def test_error_analysis_table_contains_probabilities():
    validation_data = pd.DataFrame(
        [
            {
                "chart_group_id": "group-1",
                "image_path": "chart-1.png",
                "claim_text": "Claim one",
                "label": "supported",
            },
            {
                "chart_group_id": "group-2",
                "image_path": "chart-2.png",
                "claim_text": "Claim two",
                "label": "refuted",
            },
        ]
    )

    probabilities = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.30, 0.50],
        ]
    )

    analysis = build_error_analysis_table(
        validation_data=validation_data,
        predicted_labels=[
            "supported",
            "not_enough_information",
        ],
        predicted_probabilities=(
            probabilities
        ),
        label_order=[
            "supported",
            "refuted",
            "not_enough_information",
        ],
    )

    assert analysis["correct"].tolist() == [
        True,
        False,
    ]

    assert np.isclose(
        analysis.loc[0, "confidence"],
        0.80,
    )

    assert (
        "probability_refuted"
        in analysis.columns
    )

    assert (
        analysis.loc[1, "error_type"]
        == (
            "refuted -> "
            "not_enough_information"
        )
    )


def test_error_summary_counts_only_errors():
    analysis = pd.DataFrame(
        [
            {
                "true_label": "supported",
                "predicted_label": (
                    "supported"
                ),
                "correct": True,
            },
            {
                "true_label": "refuted",
                "predicted_label": (
                    "supported"
                ),
                "correct": False,
            },
            {
                "true_label": "refuted",
                "predicted_label": (
                    "supported"
                ),
                "correct": False,
            },
        ]
    )

    summary = summarize_validation_errors(
        analysis
    )

    assert len(summary) == 1
    assert summary.loc[0, "error_count"] == 2
    assert summary.loc[0, "true_label"] == (
        "refuted"
    )
    assert summary.loc[
        0,
        "predicted_label",
    ] == "supported"
