from pathlib import Path

import pandas as pd


def test_unemployment_sample():
    project_root = Path(__file__).resolve().parents[1]
    data_path = (
        project_root
        / "data"
        / "processed"
        / "eu_unemployment_2019_2025.csv"
    )

    data = pd.read_csv(data_path)

    assert list(data.columns) == [
        "year",
        "unemployment_rate",
    ]
    assert len(data) == 7
    assert data["year"].is_unique
    assert data.isna().sum().sum() == 0
