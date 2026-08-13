from pathlib import Path

import pandas as pd

from app.ml.risk.assessment import RiskAssessmentService
from app.ml.risk.schemas import RiskPrediction


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)

FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)


def test_complete_risk_assessment_flow() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = service.assess(
        features
    )

    assert isinstance(
        result,
        RiskPrediction,
    )

    assert 0.0 <= result.risk_probability <= 1.0
    assert 0.0 <= result.risk_score <= 100.0

    expected_score = round(
        result.risk_probability * 100.0,
        2,
    )

    assert result.risk_score == expected_score

    assert result.risk_band.value in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    assert result.model_name == "xgboost"
    assert result.model_version == "1.0.0"

    assert isinstance(
        result.risk_factors,
        list,
    )

    assert len(result.risk_factors) == 5


def test_risk_factors_are_ranked() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = service.assess(
        features
    )

    importances = [
        factor.importance
        for factor in result.risk_factors
    ]

    assert importances == sorted(
        importances,
        reverse=True,
    )


def test_risk_assessment_is_repeatable() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    first = service.assess(features)
    second = service.assess(features)

    assert first.risk_probability == second.risk_probability
    assert first.risk_score == second.risk_score
    assert first.risk_band == second.risk_band
    assert first.risk_factors == second.risk_factors