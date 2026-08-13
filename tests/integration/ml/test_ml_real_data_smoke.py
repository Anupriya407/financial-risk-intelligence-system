from pathlib import Path

import pandas as pd

from app.ml.config.training import MLTrainingConfig
from app.ml.optimization.study import run_optimization


def test_real_fris_data_smoke() -> None:
    config = MLTrainingConfig()

    features_path = Path(config.features_path)
    target_path = Path(config.target_path)

    assert features_path.exists(), "FRIS feature artifact is missing."
    assert target_path.exists(), "FRIS target artifact is missing."

    features = pd.read_parquet(features_path)
    target = pd.read_parquet(target_path)["TARGET"]

    sample_size = min(5_000, len(features))

    features = features.iloc[:sample_size].reset_index(drop=True)
    target = target.iloc[:sample_size].reset_index(drop=True)

    result = run_optimization(
        X=features,
        y=target,
        n_trials=1,
        n_splits=2,
        random_state=42,
    )

    assert result.best_trial is not None
    assert result.best_value is not None
    assert 0.0 <= result.best_value <= 1.0