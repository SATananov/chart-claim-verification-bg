from src.claims import (
    make_refuted_claim,
    make_supported_claim,
    make_unknown_claim,
)


def test_supported_claim_contains_country():
    claim = make_supported_claim("Bulgaria", 2020, 2021)
    assert "Bulgaria" in claim


def test_refuted_claim_contains_years():
    claim = make_refuted_claim("Romania", 2020, 2021)
    assert "2020" in claim
    assert "2021" in claim


def test_unknown_claim_uses_future_year():
    claim = make_unknown_claim("Greece", 2027)
    assert "2027" in claim
