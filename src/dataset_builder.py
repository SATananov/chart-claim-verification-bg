from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.claims import generate_claim_dataset


CHART_STYLES = [
    {
        "style_id": "style_01",
        "marker": "o",
        "linestyle": "-",
        "figure_width": 8,
        "figure_height": 4,
        "show_grid": True,
    },
    {
        "style_id": "style_02",
        "marker": "s",
        "linestyle": "--",
        "figure_width": 7,
        "figure_height": 4,
        "show_grid": False,
    },
    {
        "style_id": "style_03",
        "marker": "^",
        "linestyle": "-.",
        "figure_width": 8,
        "figure_height": 5,
        "show_grid": True,
    },
]


def _validate_source_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "year",
        "unemployment_rate",
    }

    if not required_columns.issubset(data.columns):
        missing = required_columns.difference(
            data.columns
        )
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    clean_data = (
        data[["year", "unemployment_rate"]]
        .dropna()
        .sort_values("year")
        .reset_index(drop=True)
    )

    if len(clean_data) < 4:
        raise ValueError(
            "At least four yearly observations are required."
        )

    if not clean_data["year"].is_unique:
        raise ValueError(
            "Each year must appear only once."
        )

    return clean_data


def _save_dataset_chart(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    style: dict,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            style["figure_width"],
            style["figure_height"],
        )
    )

    axis.plot(
        data["year"],
        data["unemployment_rate"],
        marker=style["marker"],
        linestyle=style["linestyle"],
    )

    axis.set_title(title)
    axis.set_xlabel("Year")
    axis.set_ylabel("Unemployment rate (%)")

    if style["show_grid"]:
        axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_development_dataset(
    data: pd.DataFrame,
    output_directory: Path,
    relative_image_directory: str = (
        "data/generated/charts/development"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible development dataset."""

    clean_data = _validate_source_data(data)
    output_directory = Path(output_directory)

    chart_records = []
    claim_frames = []

    for window_size in range(
        4,
        len(clean_data) + 1,
    ):
        final_start_index = (
            len(clean_data) - window_size
        )

        for start_index in range(
            final_start_index + 1
        ):
            end_index = start_index + window_size

            window_data = clean_data.iloc[
                start_index:end_index
            ].copy()

            start_year = int(
                window_data["year"].iloc[0]
            )
            end_year = int(
                window_data["year"].iloc[-1]
            )

            chart_group_id = (
                f"window_{start_year}_{end_year}"
            )

            for style in CHART_STYLES:
                style_id = style["style_id"]

                chart_id = (
                    f"{chart_group_id}_{style_id}"
                )

                filename = f"{chart_id}.png"
                output_path = (
                    output_directory / filename
                )

                title = (
                    "EU Unemployment Rate, "
                    f"{start_year}-{end_year}"
                )

                _save_dataset_chart(
                    data=window_data,
                    output_path=output_path,
                    title=title,
                    style=style,
                )

                image_path = (
                    Path(relative_image_directory)
                    / filename
                ).as_posix()

                chart_records.append(
                    {
                        "chart_id": chart_id,
                        "chart_group_id": (
                            chart_group_id
                        ),
                        "style_id": style_id,
                        "window_start": start_year,
                        "window_end": end_year,
                        "number_of_years": (
                            len(window_data)
                        ),
                        "image_path": image_path,
                        "source_kind": (
                            "derived_development_sample"
                        ),
                    }
                )

                claims = generate_claim_dataset(
                    data=window_data,
                    subject=(
                        "EU unemployment rate"
                    ),
                    chart_id=chart_id,
                    unknown_year=end_year + 1,
                )

                claims.insert(
                    0,
                    "example_id",
                    [
                        (
                            f"{chart_id}_"
                            f"{claim_id}"
                        )
                        for claim_id in claims[
                            "claim_id"
                        ]
                    ],
                )

                claims["chart_group_id"] = (
                    chart_group_id
                )
                claims["style_id"] = style_id
                claims["window_start"] = (
                    start_year
                )
                claims["window_end"] = end_year
                claims["image_path"] = image_path
                claims["source_kind"] = (
                    "derived_development_sample"
                )

                claim_frames.append(claims)

    chart_manifest = pd.DataFrame(
        chart_records
    )

    claim_dataset = pd.concat(
        claim_frames,
        ignore_index=True,
    )

    claim_dataset = claim_dataset[
        [
            "example_id",
            "chart_id",
            "chart_group_id",
            "style_id",
            "window_start",
            "window_end",
            "image_path",
            "claim_id",
            "first_year",
            "second_year",
            "claim_text",
            "label",
            "claim_type",
            "source_kind",
        ]
    ]

    return chart_manifest, claim_dataset
