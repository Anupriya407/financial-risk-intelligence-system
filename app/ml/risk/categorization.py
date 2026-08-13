from enum import StrEnum


class RiskBand(StrEnum):
    """Risk classification bands."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def categorize_risk(
    probability: float,
) -> RiskBand:
    """Convert a risk probability into a risk band."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Risk probability must be between 0.0 and 1.0."
        )

    if probability < 0.30:
        return RiskBand.LOW

    if probability < 0.60:
        return RiskBand.MEDIUM

    return RiskBand.HIGH

def calculate_risk_score(
    probability: float,
) -> float:
    """Convert risk probability to a 0-100 risk score."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Risk probability must be between 0.0 and 1.0."
        )

    return round(probability * 100.0, 2)