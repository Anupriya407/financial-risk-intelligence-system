import numpy as np
import pandas as pd
import pytest

from app.ml.optimization.study import run_optimization


def test_run_optimization() -> None:
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, 5)),
        columns=[f"feature_{i}" for i in range(5)],
    )

    y = pd.Series(
        [0] * 50 + [1] * 50,
        name="target",
    )

    study = run_optimization(
        X=X,
        y=y,
        n_trials=4,
        n_splits=3,
        random_state=42,
    )

    assert len(study.trials) == 4
    assert study.best_trial is not None
    assert "model_type" in study.best_trial.params
    assert study.best_value is not None
    assert 0.0 <= study.best_value <= 1.0


def test_invalid_trial_count() -> None:
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    with pytest.raises(
        ValueError,
        match="n_trials must be at least 1",
    ):
        run_optimization(
            X=X,
            y=y,
            n_trials=0,
        )