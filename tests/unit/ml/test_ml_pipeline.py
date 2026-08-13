from pathlib import Path

import pandas as pd

from app.ml.config.training import MLTrainingConfig
from app.ml.pipeline import MLTrainingPipeline


def test_training_pipeline(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
            "feature_b": [
                12.0,
                11.0,
                10.0,
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    target = pd.DataFrame(
        {
            "TARGET": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    features_path = tmp_path / "features.parquet"
    target_path = tmp_path / "target.parquet"

    features.to_parquet(features_path)
    target.to_parquet(target_path)

    config = MLTrainingConfig(
        features_path=features_path,
        target_path=target_path,
        artifact_directory=tmp_path / "artifacts",
        n_trials=1,
        cv_folds=2,
    )

    pipeline = MLTrainingPipeline(config)

    result = pipeline.run()

    assert result.model_type in {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    }

    assert 0.0 <= result.best_score <= 1.0

    assert config.model_artifact_path().exists()
    assert config.optimization_result_path().exists()
    assert config.evaluation_result_path().exists()