import numpy as np
import pandas as pd

from app.ml.evaluation import evaluate_classifier
from app.ml.models.factory import create_model


class FakeTrial:
    def suggest_categorical(self, name, choices):
        return "logistic_regression"

    def suggest_float(self, name, *args, **kwargs):
        return 1.0

    def suggest_int(self, name, *args, **kwargs):
        return 1


def test_evaluate_classifier() -> None:
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 0.0, 1.0, 1.0],
            "feature_b": [0.0, 1.0, 0.0, 1.0],
        }
    )

    y = pd.Series([0, 0, 1, 1])

    model = create_model(FakeTrial())
    model.fit(X, y)

    metrics = evaluate_classifier(
        model,
        X,
        y,
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert all(
        0.0 <= value <= 1.0
        for value in metrics.values()
    )