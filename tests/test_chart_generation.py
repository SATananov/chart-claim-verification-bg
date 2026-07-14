from pathlib import Path

import pandas as pd

from src.chart_generation import save_line_chart


def test_save_line_chart(tmp_path: Path):
    data = pd.DataFrame(
        {
            "year": [2022, 2023, 2024],
            "unemployment_rate": [6.2, 6.1, 5.9],
        }
    )

    output_path = tmp_path / "chart.png"

    saved_path = save_line_chart(
        data=data,
        output_path=output_path,
        title="Test Chart",
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".png"
    assert saved_path.stat().st_size > 0
