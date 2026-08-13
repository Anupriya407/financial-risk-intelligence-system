from pathlib import Path

from app.ml.config.training import MLTrainingConfig


def test_default_training_config() -> None:
    config = MLTrainingConfig()

    assert config.features_path == Path(
        "data/processed/features/fris_training_features.parquet"
    )
    assert config.target_path == Path(
        "data/processed/features/fris_training_target.parquet"
    )
    assert config.test_size == 0.15
    assert config.validation_size == 0.15
    assert config.random_state == 42
    assert config.cv_folds == 5
    assert config.n_trials == 20
    assert config.optimization_metric == "roc_auc"


def test_artifact_paths() -> None:
    config = MLTrainingConfig()

    assert config.model_artifact_path() == Path(
        "data/processed/ml/best_model.joblib"
    )

    assert config.optimization_result_path() == Path(
        "data/processed/ml/optimization_result.json"
    )