from pathlib import Path
from typing import Any

import joblib
import optuna
from sklearn.pipeline import Pipeline

from app.ml.models.factory import create_model
from app.ml.optimization.results import OptimizationResult
from app.ml.preprocessing.preprocessor import MLPreprocessor


def train_best_model(
    study: optuna.Study,
    X: Any,
    y: Any,
    artifact_path: str | Path,
) -> Pipeline:
    """Train and persist the best model from an Optuna study."""

    if not study.trials:
        raise ValueError("Optuna study does not contain completed trials.")

    trial = study.best_trial

    model = _build_model_from_best_trial(trial)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                MLPreprocessor().pipeline,
            ),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifact_path)

    return pipeline


def train_best_model_from_result(
    result: OptimizationResult,
    X: Any,
    y: Any,
    artifact_path: str | Path,
) -> Pipeline:
    """Train and persist the model selected by a saved optimization result."""

    if not result.best_params:
        raise ValueError("Optimization result contains no model parameters.")

    model = _build_model_from_params(result.best_params)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                MLPreprocessor().pipeline,
            ),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifact_path)

    return pipeline


def _build_model_from_best_trial(
    trial: optuna.trial.FrozenTrial,
) -> Any:
    """Reconstruct a model from the best Optuna trial."""

    return _build_model_from_params(dict(trial.params))


def _build_model_from_params(
    params: dict[str, Any],
) -> Any:
    """Reconstruct a model directly from saved Optuna parameters."""

    class FrozenTrialAdapter:
        def __init__(self, trial_params: dict[str, Any]) -> None:
            self.params = trial_params

        def suggest_categorical(
            self,
            name: str,
            choices: list[Any],
        ) -> Any:
            return self.params[name]

        def suggest_float(
            self,
            name: str,
            *args: Any,
            **kwargs: Any,
        ) -> float:
            return self.params[name]

        def suggest_int(
            self,
            name: str,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            return self.params[name]

    return create_model(FrozenTrialAdapter(params))