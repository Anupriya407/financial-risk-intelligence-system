from pathlib import Path

import pandas as pd

from app.ml.risk.assessment import RiskAssessmentService
from app.ml.risk.schemas import RiskPrediction


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)


def test_risk_assessment_service_initializes() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    assert service.predictor is not None


def test_risk_assessment_returns_prediction() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        "data/processed/features/fris_training_features.parquet"
    ).iloc[:1]

    result = service.assess(
        features
    )

    assert isinstance(
        result,
        RiskPrediction,
    )


def test_risk_assessment_returns_valid_values() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        "data/processed/features/fris_training_features.parquet"
    ).iloc[:1]

    result = service.assess(
        features
    )

    assert 0.0 <= result.risk_probability <= 1.0
    assert 0.0 <= result.risk_score <= 100.0

    assert result.risk_band.value in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    assert result.model_name == "xgboost"
    assert result.model_version == "1.0.0"