from unittest.mock import Mock

import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from app.ml.models.factory import create_model


@pytest.mark.parametrize(
    ("model_type", "expected_class"),
    [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("xgboost", XGBClassifier),
        ("lightgbm", LGBMClassifier),
    ],
)
def test_create_model(model_type: str, expected_class: type) -> None:
    trial = Mock()

    trial.suggest_categorical.return_value = model_type
    trial.suggest_float.side_effect = lambda name, *args, **kwargs: {
        "lr_C": 1.0,
        "xgb_learning_rate": 0.1,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
        "lgbm_learning_rate": 0.1,
        "lgbm_subsample": 0.8,
        "lgbm_colsample_bytree": 0.8,
    }[name]

    trial.suggest_int.side_effect = lambda name, *args, **kwargs: {
        "rf_n_estimators": 200,
        "rf_max_depth": 10,
        "rf_min_samples_split": 2,
        "rf_min_samples_leaf": 1,
        "xgb_n_estimators": 200,
        "xgb_max_depth": 6,
        "lgbm_n_estimators": 200,
        "lgbm_num_leaves": 31,
        "lgbm_max_depth": 6,
    }[name]

    trial.suggest_categorical.side_effect = (
        lambda name, choices: (
            model_type
            if name == "model_type"
            else choices[0]
        )
    )

    model = create_model(trial)

    assert isinstance(model, expected_class)


def test_unsupported_model_is_rejected() -> None:
    trial = Mock()
    trial.suggest_categorical.return_value = "unsupported_model"

    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model(trial)