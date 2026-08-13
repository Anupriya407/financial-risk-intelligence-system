from pathlib import Path

import pandas as pd

from app.ml.risk.predictor import RiskPredictor
from app.ml.risk.schemas import RiskPrediction


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)

FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)


def test_risk_predictor_loads_model() -> None:
    predictor = RiskPredictor(
        MODEL_PATH
    )

    assert predictor.model is not None


def test_risk_predictor_returns_probability() -> None:
    predictor = RiskPredictor(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    probability = predictor.predict_probability(
        features
    )

    assert 0.0 <= probability <= 1.0


def test_risk_predictor_returns_complete_prediction() -> None:
    predictor = RiskPredictor(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = predictor.predict(
        features
    )

    assert isinstance(
        result,
        RiskPrediction,
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

    assert len(result.risk_factors) == 5

    for factor in result.risk_factors:
        assert factor.feature
        assert factor.importance >= 0.0


def test_risk_score_matches_probability() -> None:
    predictor = RiskPredictor(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = predictor.predict(
        features
    )

    expected_score = round(
        result.risk_probability * 100.0,
        2,
    )

    assert result.risk_score == expected_score