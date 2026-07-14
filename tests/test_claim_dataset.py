import pandas as pd

from src.claims import (
    VALID_LABELS,
    generate_claim_dataset,
)


def make_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [
                2019,
                2020,
                2021,
                2022,
                2023,
                2024,
                2025,
            ],
            "unemployment_rate": [
                6.8,
                7.2,
                7.1,
                6.2,
                6.1,
                5.9,
                6.0,
            ],
        }
    )


def test_claim_dataset_is_balanced():
    claims = generate_claim_dataset(
        make_sample_data()
    )

    counts = claims["label"].value_counts()

    assert len(claims) == 18
    assert set(counts.index) == set(VALID_LABELS)
    assert counts.nunique() == 1
    assert counts.iloc[0] == 6


def test_claim_dataset_has_no_missing_values():
    claims = generate_claim_dataset(
        make_sample_data()
    )

    assert claims.isna().sum().sum() == 0
    assert claims["claim_id"].is_unique
    assert claims["claim_text"].is_unique


def test_first_supported_claim_matches_data():
    claims = generate_claim_dataset(
        make_sample_data()
    )

    first_supported = claims.loc[
        claims["label"] == "supported",
        "claim_text",
    ].iloc[0]

    assert "higher in 2020 than in 2019" in first_supported


def test_unknown_year_must_be_outside_chart():
    try:
        generate_claim_dataset(
            make_sample_data(),
            unknown_year=2025,
        )
    except ValueError as error:
        assert "outside the chart range" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for an existing year."
        )
