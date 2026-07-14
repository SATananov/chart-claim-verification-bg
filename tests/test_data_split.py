import pandas as pd

from src.data_split import create_grouped_split


GROUP_SIZES = [
    27,
    27,
    27,
    27,
    36,
    36,
    36,
    45,
    45,
    54,
]


def make_grouped_data() -> pd.DataFrame:
    rows = []
    labels = [
        "supported",
        "refuted",
        "not_enough_information",
    ]

    for group_number, group_size in enumerate(
        GROUP_SIZES,
        start=1,
    ):
        group_id = (
            f"chart_group_{group_number:02d}"
        )

        for row_number in range(group_size):
            rows.append(
                {
                    "example_id": (
                        f"{group_id}_"
                        f"{row_number:03d}"
                    ),
                    "chart_group_id": group_id,
                    "label": labels[
                        row_number
                        % len(labels)
                    ],
                }
            )

    return pd.DataFrame(rows)


def test_split_row_counts_are_70_15_15():
    split_data, _ = create_grouped_split(
        make_grouped_data()
    )

    assert len(split_data["train"]) == 252
    assert len(
        split_data["validation"]
    ) == 54
    assert len(split_data["test"]) == 54


def test_groups_do_not_overlap():
    split_data, _ = create_grouped_split(
        make_grouped_data()
    )

    group_sets = {
        split_name: set(
            split_frame["chart_group_id"]
        )
        for split_name, split_frame
        in split_data.items()
    }

    assert group_sets["train"].isdisjoint(
        group_sets["validation"]
    )
    assert group_sets["train"].isdisjoint(
        group_sets["test"]
    )
    assert group_sets[
        "validation"
    ].isdisjoint(
        group_sets["test"]
    )


def test_each_split_is_label_balanced():
    split_data, _ = create_grouped_split(
        make_grouped_data()
    )

    for split_frame in split_data.values():
        counts = (
            split_frame["label"]
            .value_counts()
        )

        assert counts.nunique() == 1


def test_split_is_reproducible():
    first_split, first_manifest = (
        create_grouped_split(
            make_grouped_data(),
            random_state=42,
        )
    )

    second_split, second_manifest = (
        create_grouped_split(
            make_grouped_data(),
            random_state=42,
        )
    )

    assert first_manifest.equals(
        second_manifest
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        assert first_split[
            split_name
        ].equals(
            second_split[split_name]
        )
