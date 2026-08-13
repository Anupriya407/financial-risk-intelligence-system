from pathlib import Path

from app.ml.config.training import MLTrainingConfig
from app.ml.data_loader import MLDataLoader
from app.ml.evaluation import evaluate_classifier
from app.ml.optimization.evaluation_results import EvaluationResult
from app.ml.optimization.results import OptimizationResult
from app.ml.optimization.study import run_optimization
from app.ml.splitting.holdout import create_holdout_split
from app.ml.training import train_best_model


class MLTrainingPipeline:
    """Orchestrate the FRIS V1 machine learning training workflow."""

    def __init__(self, config: MLTrainingConfig) -> None:
        self.config = config

    def run(self) -> OptimizationResult:
        """Execute the complete training pipeline."""

        loader = MLDataLoader(
            features_path=self.config.features_path,
            target_path=self.config.target_path,
        )

        features, target = loader.load_training_data()

        if "TARGET" not in target.columns:
            raise ValueError("Target artifact must contain a TARGET column.")

        target_series = target["TARGET"]

        if len(features) != len(target_series):
            raise ValueError(
                "Features and target must contain the same number of rows."
            )

        holdout = create_holdout_split(
            features=features,
            target=target_series,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        study = run_optimization(
            X=holdout.X_development,
            y=holdout.y_development,
            n_trials=self.config.n_trials,
            n_splits=self.config.cv_folds,
            random_state=self.config.random_state,
        )

        model = train_best_model(
            study=study,
            X=holdout.X_development,
            y=holdout.y_development,
            artifact_path=self.config.model_artifact_path(),
        )

        metrics = evaluate_classifier(
            model=model,
            X_test=holdout.X_test,
            y_test=holdout.y_test,
        )

        evaluation_result = EvaluationResult(**metrics)

        evaluation_result.save(
            self.config.evaluation_result_path()
        )

        optimization_result = OptimizationResult.from_study(study)

        optimization_result.save(
            self.config.optimization_result_path()
        )

        return optimization_result