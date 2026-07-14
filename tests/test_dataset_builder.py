from pathlib import Path

import pandas as pd
import pytest

from src.dataset_builder import (
    build_development_dataset,
)


@pytest.fixture(scope="module")
def generated_dataset(
    tmp_path_factory,
):
    output_directory = (
        tmp_path_factory.mktemp("charts")
    )

    source_data = pd.DataFrame(
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

    chart_manifest, claim_dataset = (
        build_development_dataset(
            data=source_data,
            output_directory=(
                output_directory
            ),
        )
    )

    return (
        output_directory,
        chart_manifest,
        claim_dataset,
    )


def test_development_dataset_shape(
    generated_dataset,
):
    _, chart_manifest, claim_dataset = (
        generated_dataset
    )

    assert len(chart_manifest) == 30
    assert len(claim_dataset) == 360
    assert (
        chart_manifest["chart_id"].is_unique
    )
    assert (
        claim_dataset["example_id"].is_unique
    )


def test_development_dataset_is_balanced(
    generated_dataset,
):
    _, _, claim_dataset = generated_dataset

    label_counts = (
        claim_dataset["label"].value_counts()
    )

    assert label_counts.nunique() == 1
    assert label_counts.iloc[0] == 120
    assert (
        claim_dataset.isna().sum().sum()
        == 0
    )


def test_all_chart_images_are_created(
    generated_dataset,
):
    output_directory, chart_manifest, _ = (
        generated_dataset
    )

    generated_files = list(
        Path(output_directory).glob("*.png")
    )

    assert len(generated_files) == 30
    assert all(
        path.stat().st_size > 0
        for path in generated_files
    )
    assert len(chart_manifest) == len(
        generated_files
    )
