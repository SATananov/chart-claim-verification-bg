def make_supported_claim(country: str, first_year: int, second_year: int) -> str:
    return (
        f"The value for {country} was higher in "
        f"{second_year} than in {first_year}."
    )


def make_refuted_claim(country: str, first_year: int, second_year: int) -> str:
    return (
        f"The value for {country} was lower in "
        f"{second_year} than in {first_year}."
    )


def make_unknown_claim(country: str, future_year: int) -> str:
    return (
        f"The value for {country} will increase in "
        f"{future_year}."
    )
