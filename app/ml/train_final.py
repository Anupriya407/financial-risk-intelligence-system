from pathlib import Path

import pandas as pd

from app.ml.config.training import MLTrainingConfig
from app.ml.optimization.results import OptimizationResult
from app.ml.training import train_best_model_from_result


def main() -> None:
    """Train the final FRIS V1 model from the saved Optuna result."""

    config = MLTrainingConfig()

    features = pd.read_parquet(
        config.features_path
    )

    target_data = pd.read_parquet(
        config.target_path
    )

    if "TARGET" not in target_data.columns:
        raise ValueError(
            "TARGET column not found in training target dataset."
        )

    target = target_data["TARGET"]

    if len(features) != len(target):
        raise ValueError(
            "Features and target must contain the same number of rows."
        )

    result = OptimizationResult.load(
        config.optimization_result_path()
    )

    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Selected model: {result.model_type}")
    print(f"Best CV ROC-AUC: {result.best_score:.6f}")
    print()
    print("Training final model...")

    train_best_model_from_result(
        result=result,
        X=features,
        y=target,
        artifact_path=config.model_artifact_path(),
    )

    print()
    print("=" * 60)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {result.model_type}")
    print(f"CV ROC-AUC: {result.best_score:.6f}")
    print(
        f"Model artifact: "
        f"{config.model_artifact_path()}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()