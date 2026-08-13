import pandas as pd
import pytest

from app.ml.risk.predictor import RiskPredictor


MODEL_PATH = "data/processed/ml/best_model.joblib"
FEATURES_PATH = (
    "data/processed/features/fris_training_features.parquet"
)


def test_predictor_rejects_multiple_rows() -> None:
    predictor = RiskPredictor(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:2]

    with pytest.raises(ValueError):
        predictor.predict_probability(
            features
        )