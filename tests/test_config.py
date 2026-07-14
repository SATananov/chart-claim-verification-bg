from src.config import LABELS


def test_project_has_three_labels():
    assert LABELS == [
        "supported",
        "refuted",
        "not_enough_information",
    ]
