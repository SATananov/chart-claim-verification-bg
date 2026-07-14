import pandas as pd

from src.baselines import (
    build_majority_baseline,
    build_text_baseline,
    evaluate_classifier,
)


def make_training_data():
    text_values = [
        "chart says increase supported",
        "chart says decrease refuted",
        "future year unknown",
    ] * 4

    labels = [
        "supported",
        "refuted",
        "not_enough_information",
    ] * 4

    return text_values, labels


def test_majority_baseline_predicts_one_class():
    text_values, labels = make_training_data()

    model = build_majority_baseline()
    model.fit(
        pd.DataFrame(
            {"claim_text": text_values}
        ),
        labels,
    )

    predictions = model.predict(
        pd.DataFrame(
            {"claim_text": text_values}
        )
    )

    assert len(set(predictions)) == 1


def test_text_baseline_learns_simple_patterns():
    text_values, labels = make_training_data()

    model = build_text_baseline()
    model.fit(text_values, labels)

    predictions = model.predict(
        [
            "supported increase",
            "refuted decrease",
            "unknown future",
        ]
    )

    assert list(predictions) == [
        "supported",
        "refuted",
        "not_enough_information",
    ]


def test_evaluation_returns_expected_tables():
    text_values, labels = make_training_data()

    model = build_text_baseline()
    model.fit(text_values, labels)

    metrics, report, matrix = (
        evaluate_classifier(
            model,
            text_values,
            labels,
        )
    )

    assert set(metrics) == {
        "accuracy",
        "macro_f1",
    }
    assert "macro avg" in report.index
    assert matrix.shape == (3, 3)
