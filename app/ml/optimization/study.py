from typing import Any

import optuna
import pandas as pd

from app.ml.optimization.objective import create_objective
from app.ml.optimization.results import OptimizationResult


def run_optimization(
    X: Any,
    y: Any,
    n_trials: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
) -> optuna.Study:
    """Run Optuna model selection and hyperparameter optimization."""

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    sampler = optuna.samplers.TPESampler(
        seed=random_state,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    objective = create_objective(
        X=X,
        y=y,
        n_splits=n_splits,
        random_state=random_state,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    return study


def main() -> None:
    """Run the FRIS Optuna optimization from the command line."""

    features_path = (
        "data/processed/features/fris_training_features.parquet"
    )
    target_path = (
        "data/processed/features/fris_training_target.parquet"
    )

    X = pd.read_parquet(features_path)
    target_df = pd.read_parquet(target_path)

    if "TARGET" not in target_df.columns:
        raise ValueError("TARGET column not found in training target dataset.")

    y = target_df["TARGET"]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("Starting Optuna optimization...")
    print("Trials: 20")
    print("CV folds: 5")
    print("Metric: ROC-AUC")
    print()

    study = run_optimization(
        X=X,
        y=y,
        n_trials=20,
        n_splits=5,
        random_state=42,
    )
    result = OptimizationResult.from_study(study)

    result_path = "data/processed/ml/optimization_result.json"
    result.save(result_path)

    print(f"Optimization result saved to: {result_path}")

    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best ROC-AUC: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    print("=" * 60)


if __name__ == "__main__":
    main()