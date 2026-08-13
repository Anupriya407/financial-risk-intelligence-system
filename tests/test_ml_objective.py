import numpy as np
import optuna
import pandas as pd

from app.ml.optimization.objective import create_objective


def test_objective_runs_cross_validation() -> None:
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, 5)),
        columns=[f"feature_{i}" for i in range(5)],
    )

    y = pd.Series(
        np.array([0] * 50 + [1] * 50),
        name="target",
    )

    study = optuna.create_study(direction="maximize")

    study.optimize(
        create_objective(
            X=X,
            y=y,
            n_splits=5,
            random_state=42,
        ),
        n_trials=1,
    )

    assert study.best_value is not None
    assert 0.0 <= study.best_value <= 1.0

    assert "cv_scores" in study.best_trial.user_attrs
    assert len(study.best_trial.user_attrs["cv_scores"]) == 5


def test_objective_rejects_invalid_fold_count() -> None:
    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [4, 3, 2, 1],
        }
    )

    y = pd.Series([0, 0, 1, 1])

    try:
        create_objective(
            X=X,
            y=y,
            n_splits=1,
        )
    except ValueError as exc:
        assert str(exc) == "n_splits must be at least 2."
    else:
        raise AssertionError("Expected ValueError")