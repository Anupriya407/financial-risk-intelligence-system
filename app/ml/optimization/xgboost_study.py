from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.ml.preprocessing.preprocessor import MLPreprocessor


FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)

TARGET_PATH = Path(
    "data/processed/features/fris_training_target.parquet"
)

RESULT_PATH = Path(
    "data/processed/ml/xgboost_optimization_result.json"
)


def create_xgboost_objective(
    X: Any,
    y: Any,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Create an XGBoost-only Optuna objective."""

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_int(
                "xgb_n_estimators",
                200,
                800,
            ),
            max_depth=trial.suggest_int(
                "xgb_max_depth",
                3,
                12,
            ),
            learning_rate=trial.suggest_float(
                "xgb_learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            subsample=trial.suggest_float(
                "xgb_subsample",
                0.6,
                1.0,
            ),
            colsample_bytree=trial.suggest_float(
                "xgb_colsample_bytree",
                0.6,
                1.0,
            ),
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

        fold_scores: list[float] = []

        for train_idx, validation_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            X_validation = X.iloc[validation_idx]

            y_train = y.iloc[train_idx]
            y_validation = y.iloc[validation_idx]

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        clone(
                            MLPreprocessor().pipeline
                        ),
                    ),
                    (
                        "model",
                        clone(model),
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            probabilities = pipeline.predict_proba(
                X_validation
            )[:, 1]

            score = roc_auc_score(
                y_validation,
                probabilities,
            )

            fold_scores.append(
                float(score)
            )

        mean_score = float(
            np.mean(fold_scores)
        )

        trial.set_user_attr(
            "cv_scores",
            fold_scores,
        )

        trial.set_user_attr(
            "cv_mean_roc_auc",
            mean_score,
        )

        return mean_score

    return objective


def main() -> None:
    """Run XGBoost-only Optuna optimization."""

    features = pd.read_parquet(
        FEATURES_PATH
    )

    target_df = pd.read_parquet(
        TARGET_PATH
    )

    if "TARGET" not in target_df.columns:
        raise ValueError(
            "TARGET column not found."
        )

    target = target_df["TARGET"]

    if len(features) != len(target):
        raise ValueError(
            "Features and target must have "
            "the same number of rows."
        )

    print(
        f"Features shape: {features.shape}"
    )

    print(
        f"Target shape: {target.shape}"
    )

    print()
    print(
        "Starting XGBoost-only Optuna optimization..."
    )
    print("Trials: 20")
    print("CV folds: 5")
    print()

    sampler = optuna.samplers.TPESampler(
        seed=42,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    objective = create_xgboost_objective(
        X=features,
        y=target,
        n_splits=5,
        random_state=42,
    )

    study.optimize(
        objective,
        n_trials=20,
    )

    best_trial = study.best_trial

    print()
    print("=" * 60)
    print("XGBOOST OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(
        f"Best XGBoost ROC-AUC: "
        f"{study.best_value:.6f}"
    )
    print(
        f"Best parameters: "
        f"{best_trial.params}"
    )
    print("=" * 60)

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "model_type": "xgboost",
        "best_score": float(
            study.best_value
        ),
        "best_params": {
            "model_type": "xgboost",
            **best_trial.params,
        },
        "n_trials": len(study.trials),
        "cv_scores": [
            float(score)
            for score in best_trial.user_attrs.get(
                "cv_scores",
                [],
            )
        ],
    }

    import json

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Result saved to: {RESULT_PATH}"
    )

    print()
    print("MODEL COMPARISON")
    print("-" * 40)
    print(
        "LightGBM ROC-AUC: 0.786710"
    )
    print(
        f"XGBoost ROC-AUC:  "
        f"{study.best_value:.6f}"
    )

    difference = (
        study.best_value - 0.786710
    )

    print(
        f"Difference:       "
        f"{difference:+.6f}"
    )

    if difference > 0:
        print("Winner: XGBoost")
    else:
        print("Winner: LightGBM")


if __name__ == "__main__":
    main()