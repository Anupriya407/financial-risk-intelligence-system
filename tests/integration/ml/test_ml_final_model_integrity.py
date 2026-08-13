from pathlib import Path

import joblib
from xgboost import XGBClassifier


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)


def test_final_artifact_is_pipeline() -> None:
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "named_steps")
    assert "preprocessor" in model.named_steps
    assert "model" in model.named_steps


def test_final_artifact_uses_xgboost() -> None:
    model = joblib.load(MODEL_PATH)

    estimator = model.named_steps["model"]

    assert isinstance(
        estimator,
        XGBClassifier,
    )


def test_final_artifact_matches_optimized_parameters() -> None:
    model = joblib.load(MODEL_PATH)

    estimator = model.named_steps["model"]

    assert estimator.n_estimators == 671
    assert estimator.max_depth == 4
    assert estimator.learning_rate == 0.05748924681991978
    assert estimator.subsample == 0.836965827544817
    assert estimator.colsample_bytree == 0.6185801650879991