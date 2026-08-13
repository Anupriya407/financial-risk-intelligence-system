import pytest
from pydantic import ValidationError

from app.api.v1.schemas.risk_prediction import (
    RiskPredictionRequest,
    RiskPredictionResponse,
)
from app.ml.risk.categorization import RiskBand
from app.ml.risk.schemas import RiskPrediction


def test_risk_prediction_request_accepts_features() -> None:
    request = RiskPredictionRequest(
        features={
            "feature_a": 10.0,
            "feature_b": 20.0,
        }
    )

    assert request.features["feature_a"] == 10.0
    assert request.features["feature_b"] == 20.0


def test_risk_prediction_request_requires_features() -> None:
    with pytest.raises(ValidationError):
        RiskPredictionRequest(
            features={}
        )


def test_risk_prediction_response_accepts_prediction() -> None:
    prediction = RiskPrediction(
        risk_probability=0.75,
        risk_score=75.0,
        risk_band=RiskBand.HIGH,
        model_name="xgboost",
        model_version="1.0.0",
    )

    response = RiskPredictionResponse(
        prediction=prediction
    )

    assert response.prediction == prediction