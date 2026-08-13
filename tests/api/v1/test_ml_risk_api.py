from pathlib import Path

import pandas as pd
import pytest

from app.ml.risk.assessment import RiskAssessmentService


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)

FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)


def test_assessment_service_returns_prediction() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = service.assess(features)

    assert result.risk_probability >= 0.0
    assert result.risk_probability <= 1.0

    assert result.risk_score >= 0.0
    assert result.risk_score <= 100.0

    assert result.risk_band.value in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    assert result.model_name == "xgboost"
    assert result.model_version == "1.0.0"


def test_assessment_service_returns_serializable_result() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = service.assess(features)

    payload = result.model_dump()

    assert isinstance(payload, dict)

    assert "risk_probability" in payload
    assert "risk_score" in payload
    assert "risk_band" in payload
    assert "model_name" in payload
    assert "model_version" in payload
    assert "risk_factors" in payload


def test_invalid_input_is_rejected() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    with pytest.raises(TypeError):
        service.assess(None)