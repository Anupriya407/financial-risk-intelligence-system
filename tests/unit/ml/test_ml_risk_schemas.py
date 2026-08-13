import pytest
from pydantic import ValidationError

from app.ml.risk.categorization import RiskBand
from app.ml.risk.schemas import RiskFactor, RiskPrediction


def test_valid_risk_prediction() -> None:
    prediction = RiskPrediction(
        risk_probability=0.75,
        risk_score=75.0,
        risk_band=RiskBand.HIGH,
        model_name="xgboost",
        model_version="1.0.0",
        risk_factors=[
            {
                "feature": "feature_a",
                "importance": 0.42,
            },
        ],
    )

    assert prediction.risk_probability == 0.75
    assert prediction.risk_score == 75.0
    assert prediction.risk_band == RiskBand.HIGH
    assert prediction.model_name == "xgboost"
    assert prediction.model_version == "1.0.0"
    assert len(prediction.risk_factors) == 1


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01],
)
def test_invalid_probability(
    probability: float,
) -> None:
    with pytest.raises(ValidationError):
        RiskPrediction(
            risk_probability=probability,
            risk_score=50.0,
            risk_band=RiskBand.MEDIUM,
            model_name="xgboost",
            model_version="1.0.0",
        )


@pytest.mark.parametrize(
    "score",
    [-1.0, 100.01],
)
def test_invalid_risk_score(
    score: float,
) -> None:
    with pytest.raises(ValidationError):
        RiskPrediction(
            risk_probability=0.5,
            risk_score=score,
            risk_band=RiskBand.MEDIUM,
            model_name="xgboost",
            model_version="1.0.0",
        )


def test_risk_band_is_required() -> None:
    with pytest.raises(ValidationError):
        RiskPrediction(
            risk_probability=0.5,
            risk_score=50.0,
            model_name="xgboost",
            model_version="1.0.0",
        )


def test_model_metadata_is_required() -> None:
    with pytest.raises(ValidationError):
        RiskPrediction(
            risk_probability=0.5,
            risk_score=50.0,
            risk_band=RiskBand.MEDIUM,
        )


def test_inconsistent_risk_score_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskPrediction(
            risk_probability=0.75,
            risk_score=25.0,
            risk_band=RiskBand.HIGH,
            model_name="xgboost",
            model_version="1.0.0",
        )


def test_inconsistent_risk_band_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskPrediction(
            risk_probability=0.75,
            risk_score=75.0,
            risk_band=RiskBand.LOW,
            model_name="xgboost",
            model_version="1.0.0",
        )


def test_risk_factors_default_to_empty_list() -> None:
    prediction = RiskPrediction(
        risk_probability=0.75,
        risk_score=75.0,
        risk_band=RiskBand.HIGH,
        model_name="xgboost",
        model_version="1.0.0",
    )

    assert prediction.risk_factors == []

def test_risk_factor_schema() -> None:
    factor = RiskFactor(
        feature="credit_utilization",
        importance=0.42,
    )

    assert factor.feature == "credit_utilization"
    assert factor.importance == 0.42

def test_risk_factor_rejects_empty_feature() -> None:
    with pytest.raises(ValidationError):
        RiskFactor(
            feature="",
            importance=0.42,
        )


def test_risk_factor_rejects_negative_importance() -> None:
    with pytest.raises(ValidationError):
        RiskFactor(
            feature="credit_utilization",
            importance=-0.1,
        )