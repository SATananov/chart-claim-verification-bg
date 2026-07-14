from __future__ import annotations

import pandas as pd


VALID_LABELS = [
    "supported",
    "refuted",
    "not_enough_information",
]


def make_supported_claim(
    subject: str,
    first_year: int,
    second_year: int,
) -> str:
    """Create a simple higher-value claim."""

    return (
        f"The {subject} was higher in "
        f"{second_year} than in {first_year}."
    )


def make_refuted_claim(
    subject: str,
    first_year: int,
    second_year: int,
) -> str:
    """Create a simple lower-value claim."""

    return (
        f"The {subject} was lower in "
        f"{second_year} than in {first_year}."
    )


def make_unknown_claim(
    subject: str,
    future_year: int,
) -> str:
    """Create a claim about a year outside the chart."""

    return (
        f"The {subject} will increase in "
        f"{future_year}."
    )


def _comparison_claim(
    subject: str,
    first_year: int,
    second_year: int,
    relation: str,
) -> str:
    return (
        f"The {subject} was {relation} in "
        f"{second_year} than in {first_year}."
    )


def generate_claim_dataset(
    data: pd.DataFrame,
    subject: str = "EU unemployment rate",
    chart_id: str = "eu_unemployment_2019_2025",
    unknown_year: int = 2026,
) -> pd.DataFrame:
    """Generate balanced claims from adjacent yearly values."""

    required_columns = {
        "year",
        "unemployment_rate",
    }

    if not required_columns.issubset(data.columns):
        missing = required_columns.difference(data.columns)
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    ordered_data = (
        data[["year", "unemployment_rate"]]
        .sort_values("year")
        .reset_index(drop=True)
    )

    if len(ordered_data) < 2:
        raise ValueError(
            "At least two yearly observations are required."
        )

    if unknown_year in set(ordered_data["year"]):
        raise ValueError(
            "The unknown year must be outside the chart range."
        )

    records = []
    claim_number = 1

    for index in range(len(ordered_data) - 1):
        first_row = ordered_data.iloc[index]
        second_row = ordered_data.iloc[index + 1]

        first_year = int(first_row["year"])
        second_year = int(second_row["year"])
        first_value = float(first_row["unemployment_rate"])
        second_value = float(second_row["unemployment_rate"])

        if second_value > first_value:
            true_relation = "higher"
            false_relation = "lower"
        elif second_value < first_value:
            true_relation = "lower"
            false_relation = "higher"
        else:
            true_relation = "the same"
            false_relation = "different"

        pair_claims = [
            {
                "claim_text": _comparison_claim(
                    subject,
                    first_year,
                    second_year,
                    true_relation,
                ),
                "label": "supported",
                "claim_type": "true_comparison",
            },
            {
                "claim_text": _comparison_claim(
                    subject,
                    first_year,
                    second_year,
                    false_relation,
                ),
                "label": "refuted",
                "claim_type": "false_comparison",
            },
            {
                "claim_text": (
                    f"The {subject} was higher in "
                    f"{unknown_year} than in {second_year}."
                ),
                "label": "not_enough_information",
                "claim_type": "outside_chart_range",
            },
        ]

        for claim in pair_claims:
            records.append(
                {
                    "claim_id": f"claim_{claim_number:03d}",
                    "chart_id": chart_id,
                    "first_year": first_year,
                    "second_year": second_year,
                    **claim,
                }
            )
            claim_number += 1

    claim_data = pd.DataFrame(records)

    return claim_data[
        [
            "claim_id",
            "chart_id",
            "first_year",
            "second_year",
            "claim_text",
            "label",
            "claim_type",
        ]
    ]
