from typing import Any

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from app.ml.models.factory import create_model
from app.ml.preprocessing.preprocessor import MLPreprocessor


def create_objective(
    X: Any,
    y: Any,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Create an Optuna objective using leakage-safe cross-validation."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        model = create_model(trial)

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
                        clone(MLPreprocessor().pipeline),
                    ),
                    ("model", clone(model)),
                ]
            )

            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(
                X_validation
            )[:, 1]

            score = roc_auc_score(
                y_validation,
                probabilities,
            )

            fold_scores.append(float(score))

        mean_score = float(np.mean(fold_scores))

        trial.set_user_attr("cv_scores", fold_scores)
        trial.set_user_attr("cv_mean_roc_auc", mean_score)

        return mean_score

    return objective