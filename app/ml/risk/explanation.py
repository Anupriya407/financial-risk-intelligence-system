from typing import Any
from app.ml.risk.schemas import RiskFactor
import numpy as np


def extract_risk_factors(
    model: Any,
    features: Any,
    top_n: int = 5,
) -> list[RiskFactor]:
    """Extract the strongest feature contributions for an XGBoost model."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if not hasattr(model, "named_steps"):
        raise TypeError(
            "Expected a fitted sklearn Pipeline."
        )

    if "preprocessor" not in model.named_steps:
        raise ValueError(
            "Pipeline does not contain a preprocessor."
        )

    if "model" not in model.named_steps:
        raise ValueError(
            "Pipeline does not contain a model."
        )

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        raise TypeError(
            "Model does not expose feature_importances_."
        )

    transformed_features = preprocessor.transform(
        features
    )

    feature_names = preprocessor.get_feature_names_out()

    importances = np.asarray(
        estimator.feature_importances_
    )

    if transformed_features.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "Feature names and transformed features "
            "have inconsistent dimensions."
        )

    if len(importances) != len(feature_names):
        raise ValueError(
            "Model feature importances and transformed "
            "features have inconsistent dimensions."
        )

    ranked_indices = np.argsort(
        importances
    )[::-1][:top_n]

    return [
    RiskFactor(
        feature=str(feature_names[index]),
        importance=float(importances[index]),
    )
    for index in ranked_indices
    ]