from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLTrainingConfig:
    """Configuration for the FRIS V1 ML training pipeline."""

    features_path: Path = Path(
        "data/processed/features/fris_training_features.parquet"
    )
    target_path: Path = Path(
        "data/processed/features/fris_training_target.parquet"
    )
    artifact_directory: Path = Path("data/processed/ml")

    model_artifact_name: str = "best_model.joblib"
    optimization_result_name: str = "optimization_result.json"
    evaluation_result_name: str = "evaluation_metrics.json"

    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42

    cv_folds: int = 5
    n_trials: int = 20

    optimization_metric: str = "roc_auc"

    def model_artifact_path(self) -> Path:
        """Return the trained model artifact path."""
        return self.artifact_directory / self.model_artifact_name

    def optimization_result_path(self) -> Path:
        """Return the optimization result path."""
        return self.artifact_directory / self.optimization_result_name

    def evaluation_result_path(self) -> Path:
        """Return the final evaluation result path."""
        return self.artifact_directory / self.evaluation_result_name