import json
from pathlib import Path

import optuna

from app.ml.optimization.results import OptimizationResult


def test_result_from_study() -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical(
            "model_type",
            ["logistic_regression"],
        )

        trial.set_user_attr(
            "cv_scores",
            [0.71, 0.73, 0.72],
        )

        return 0.72

    study.optimize(objective, n_trials=1)

    result = OptimizationResult.from_study(study)

    assert result.model_type == "logistic_regression"
    assert result.best_score == 0.72
    assert result.n_trials == 1
    assert result.cv_scores == [0.71, 0.73, 0.72]
    assert "model_type" in result.best_params
    assert result.completed_at


def test_result_save_and_load(tmp_path: Path) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_categorical(
            "model_type",
            ["logistic_regression"],
        )

        trial.set_user_attr(
            "cv_scores",
            [0.75, 0.76],
        )

        return 0.755

    study.optimize(objective, n_trials=1)

    result = OptimizationResult.from_study(study)

    path = tmp_path / "optimization_result.json"

    result.save(path)

    assert path.exists()

    loaded = OptimizationResult.load(path)

    assert loaded == result


def test_saved_result_is_valid_json(tmp_path: Path) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_categorical(
            "model_type",
            ["logistic_regression"],
        )

        return 0.80

    study.optimize(objective, n_trials=1)

    result = OptimizationResult.from_study(study)

    path = tmp_path / "result.json"

    result.save(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert data["model_type"] == "logistic_regression"
    assert data["best_score"] == 0.80