from __future__ import annotations

import numpy as np
import pandas as pd


SPLIT_NAMES = [
    "train",
    "validation",
    "test",
]


def _check_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = [
        train_ratio,
        validation_ratio,
        test_ratio,
    ]

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(
            "All split ratios must be positive."
        )

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(
            "The split ratios must sum to 1.0."
        )


def _choose_group_assignment(
    group_sizes: pd.Series,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, str]:
    """Choose a deterministic assignment for a small group set."""

    group_ids = group_sizes.index.to_list()

    if len(group_ids) < 3:
        raise ValueError(
            "At least three groups are required."
        )

    rng = np.random.default_rng(random_state)
    rng.shuffle(group_ids)

    sizes = {
        group_id: int(group_sizes.loc[group_id])
        for group_id in group_ids
    }

    total_rows = sum(sizes.values())

    targets = {
        "train": total_rows * train_ratio,
        "validation": (
            total_rows * validation_ratio
        ),
        "test": total_rows * test_ratio,
    }

    # State: (validation rows, test rows) -> assignments so far.
    states = {
        (0, 0): tuple()
    }

    for group_id in group_ids:
        group_size = sizes[group_id]
        next_states = {}

        for (
            validation_rows,
            test_rows,
        ), assignment in states.items():
            choices = [
                (
                    validation_rows,
                    test_rows,
                    "train",
                ),
                (
                    validation_rows
                    + group_size,
                    test_rows,
                    "validation",
                ),
                (
                    validation_rows,
                    test_rows + group_size,
                    "test",
                ),
            ]

            for (
                new_validation_rows,
                new_test_rows,
                split_name,
            ) in choices:
                state_key = (
                    new_validation_rows,
                    new_test_rows,
                )

                if state_key not in next_states:
                    next_states[state_key] = (
                        assignment
                        + ((group_id, split_name),)
                    )

        states = next_states

    best_score = None
    best_assignment = None

    for (
        validation_rows,
        test_rows,
    ), assignment in states.items():
        split_by_group = dict(assignment)

        split_group_counts = {
            split_name: sum(
                assigned_split == split_name
                for assigned_split
                in split_by_group.values()
            )
            for split_name in SPLIT_NAMES
        }

        if any(
            count == 0
            for count in split_group_counts.values()
        ):
            continue

        train_rows = (
            total_rows
            - validation_rows
            - test_rows
        )

        row_counts = {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        }

        row_error = sum(
            abs(
                row_counts[split_name]
                - targets[split_name]
            )
            for split_name in SPLIT_NAMES
        )

        group_error = sum(
            abs(
                split_group_counts[split_name]
                - (
                    len(group_ids)
                    * {
                        "train": train_ratio,
                        "validation": validation_ratio,
                        "test": test_ratio,
                    }[split_name]
                )
            )
            for split_name in SPLIT_NAMES
        )

        score = (
            round(row_error, 10),
            round(group_error, 10),
        )

        if (
            best_score is None
            or score < best_score
        ):
            best_score = score
            best_assignment = split_by_group

    if best_assignment is None:
        raise RuntimeError(
            "A valid group split was not found."
        )

    return best_assignment


def create_grouped_split(
    data: pd.DataFrame,
    group_column: str = "chart_group_id",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Split rows while keeping every group in one set."""

    _check_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if group_column not in data.columns:
        raise ValueError(
            f"Missing group column: {group_column}"
        )

    if data[group_column].isna().any():
        raise ValueError(
            "The group column contains missing values."
        )

    group_sizes = (
        data.groupby(group_column)
        .size()
        .sort_index()
    )

    assignment = _choose_group_assignment(
        group_sizes=group_sizes,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    data_with_split = data.copy()

    data_with_split["split"] = (
        data_with_split[group_column]
        .map(assignment)
    )

    split_data = {
        split_name: (
            data_with_split.loc[
                data_with_split["split"]
                == split_name
            ]
            .reset_index(drop=True)
        )
        for split_name in SPLIT_NAMES
    }

    manifest = (
        group_sizes
        .rename("row_count")
        .reset_index()
    )

    manifest["split"] = (
        manifest[group_column]
        .map(assignment)
    )

    manifest = manifest.sort_values(
        ["split", group_column]
    ).reset_index(drop=True)

    group_sets = {
        split_name: set(
            split_frame[group_column]
        )
        for split_name, split_frame
        in split_data.items()
    }

    if (
        group_sets["train"]
        & group_sets["validation"]
        or group_sets["train"]
        & group_sets["test"]
        or group_sets["validation"]
        & group_sets["test"]
    ):
        raise RuntimeError(
            "Group leakage was detected."
        )

    if sum(
        len(split_frame)
        for split_frame in split_data.values()
    ) != len(data):
        raise RuntimeError(
            "Some rows were lost during the split."
        )

    return split_data, manifest
