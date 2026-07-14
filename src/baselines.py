from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


LABEL_ORDER = [
    "supported",
    "refuted",
    "not_enough_information",
]


def build_majority_baseline() -> DummyClassifier:
    """Create a majority-class classifier."""

    return DummyClassifier(
        strategy="most_frequent",
    )


def build_text_baseline(
    random_state: int = 42,
) -> Pipeline:
    """Create a TF-IDF and logistic regression pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_classifier(
    model,
    text_values,
    true_labels,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Return summary metrics, report and confusion matrix."""

    predictions = model.predict(text_values)

    metrics = {
        "accuracy": accuracy_score(
            true_labels,
            predictions,
        ),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    report = pd.DataFrame(
        classification_report(
            true_labels,
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    matrix = pd.DataFrame(
        confusion_matrix(
            true_labels,
            predictions,
            labels=LABEL_ORDER,
        ),
        index=[
            f"actual_{label}"
            for label in LABEL_ORDER
        ],
        columns=[
            f"predicted_{label}"
            for label in LABEL_ORDER
        ],
    )

    return metrics, report, matrix
