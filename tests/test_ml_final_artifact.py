from pathlib import Path

import joblib


ARTIFACT_PATH = Path(
    "data/processed/ml/best_model.joblib"
)


def test_final_model_artifact_exists() -> None:
    assert ARTIFACT_PATH.exists()
    assert ARTIFACT_PATH.is_file()


def test_final_model_artifact_loads() -> None:
    model = joblib.load(ARTIFACT_PATH)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_final_model_contains_preprocessor_and_model() -> None:
    model = joblib.load(ARTIFACT_PATH)

    assert hasattr(model, "named_steps")

    assert "preprocessor" in model.named_steps
    assert "model" in model.named_steps


def test_final_model_is_xgboost() -> None:
    model = joblib.load(ARTIFACT_PATH)

    estimator = model.named_steps["model"]

    assert estimator.__class__.__name__ == "XGBClassifier"