import pytest

from app.ml.risk.categorization import (
    RiskBand,
    calculate_risk_score,
    categorize_risk,
)


def test_low_risk_probability() -> None:
    assert categorize_risk(0.10) == RiskBand.LOW


def test_medium_risk_probability() -> None:
    assert categorize_risk(0.45) == RiskBand.MEDIUM


def test_high_risk_probability() -> None:
    assert categorize_risk(0.80) == RiskBand.HIGH


def test_lower_boundary() -> None:
    assert categorize_risk(0.30) == RiskBand.MEDIUM


def test_upper_boundary() -> None:
    assert categorize_risk(0.60) == RiskBand.HIGH


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01],
)
def test_invalid_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        categorize_risk(probability)

@pytest.mark.parametrize(
    ("probability", "expected_score"),
    [
        (0.0, 0.0),
        (0.1234, 12.34),
        (0.5, 50.0),
        (0.8765, 87.65),
        (1.0, 100.0),
    ],
)
def test_calculate_risk_score(
    probability: float,
    expected_score: float,
) -> None:
    assert (
        calculate_risk_score(probability)
        == expected_score
    )