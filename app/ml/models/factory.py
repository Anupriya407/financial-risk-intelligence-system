from typing import Any

import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


SUPPORTED_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
)


def create_model(trial: optuna.Trial) -> Any:
    """Create an ML model using Optuna-selected hyperparameters."""

    model_type = trial.suggest_categorical(
        "model_type",
        SUPPORTED_MODELS,
    )

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("lr_C", 1e-3, 10.0, log=True),
            max_iter=1000,
            random_state=42,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 200, 800),
            max_depth=trial.suggest_int("rf_max_depth", 5, 30),
            min_samples_split=trial.suggest_int(
                "rf_min_samples_split",
                2,
                20,
            ),
            min_samples_leaf=trial.suggest_int(
                "rf_min_samples_leaf",
                1,
                10,
            ),
            max_features=trial.suggest_categorical(
                "rf_max_features",
                ["sqrt", "log2"],
            ),
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 800),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 12),
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
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )

    if model_type == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("lgbm_n_estimators", 200, 800),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 127),
            max_depth=trial.suggest_int("lgbm_max_depth", 3, 15),
            learning_rate=trial.suggest_float(
                "lgbm_learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            subsample=trial.suggest_float(
                "lgbm_subsample",
                0.6,
                1.0,
            ),
            colsample_bytree=trial.suggest_float(
                "lgbm_colsample_bytree",
                0.6,
                1.0,
            ),
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported model type: {model_type}")