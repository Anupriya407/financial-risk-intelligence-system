from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from app.ml.optimization.results import OptimizationResult
from app.ml.training import (
    train_best_model,
    train_best_model_from_result,
)


def test_train_best_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(80, 5)),
        columns=[f"feature_{i}" for i in range(5)],
    )

    y = pd.Series(
        [0] * 40 + [1] * 40,
        name="target",
    )

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical(
            "model_type",
            ["logistic_regression"],
        )

        assert model_type == "logistic_regression"

        c_value = trial.suggest_float(
            "lr_C",
            0.1,
            10.0,
        )

        return 0.8 if c_value > 1.0 else 0.7

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=2,
    )

    artifact_path = tmp_path / "best_model.joblib"

    model = train_best_model(
        study=study,
        X=X,
        y=y,
        artifact_path=artifact_path,
    )

    assert model is not None
    assert artifact_path.exists()

    loaded_model = __import__("joblib").load(
        artifact_path,
    )

    predictions = loaded_model.predict(X)

    assert len(predictions) == len(X)


def test_train_best_model_from_saved_result(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(80, 5)),
        columns=[f"feature_{i}" for i in range(5)],
    )

    y = pd.Series(
        [0] * 40 + [1] * 40,
        name="target",
    )

    result = OptimizationResult(
        model_type="logistic_regression",
        best_score=0.80,
        best_params={
            "model_type": "logistic_regression",
            "lr_C": 2.0,
        },
        n_trials=2,
        cv_scores=[0.78, 0.82],
        completed_at="2026-08-11T00:00:00+00:00",
    )

    artifact_path = tmp_path / "best_model.joblib"

    model = train_best_model_from_result(
        result=result,
        X=X,
        y=y,
        artifact_path=artifact_path,
    )

    assert model is not None
    assert artifact_path.exists()

    loaded_model = __import__("joblib").load(
        artifact_path,
    )

    predictions = loaded_model.predict(X)

    assert len(predictions) == len(X)