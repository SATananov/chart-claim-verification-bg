from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_line_chart(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    x_column: str = "year",
    y_column: str = "unemployment_rate",
) -> Path:
    """Create and save a simple line chart."""

    required_columns = {x_column, y_column}

    if not required_columns.issubset(data.columns):
        missing = required_columns.difference(data.columns)
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4))

    axis.plot(
        data[x_column],
        data[y_column],
        marker="o",
    )

    axis.set_title(title)
    axis.set_xlabel("Year")
    axis.set_ylabel("Unemployment rate (%)")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path
