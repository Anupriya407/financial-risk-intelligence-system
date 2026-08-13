from typing import Any

import pandas as pd

from app.ml.risk.predictor import RiskPredictor
from app.ml.risk.schemas import RiskPrediction


class RiskAssessmentService:
    """Service for generating complete FRIS risk assessments."""

    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.predictor = RiskPredictor(
            model_path
        )

    def assess(
        self,
        features: Any,
    ) -> RiskPrediction:
        """Generate a complete financial risk assessment."""

        self._validate_features(features)

        return self.predictor.predict(
            features
        )

    def _validate_features(
        self,
        features: Any,
    ) -> None:
        """Validate the input feature matrix before prediction."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if features.empty:
            raise ValueError(
                "features must contain at least one row."
            )

        if len(features) != 1:
            raise ValueError(
                "Risk assessment requires exactly one customer row."
            )

        model_features = getattr(
            self.predictor.model,
            "feature_names_in_",
            None,
        )

        if model_features is None:
            raise ValueError(
                "Unable to determine the model's expected features."
            )

        expected_features = list(
            model_features
        )

        actual_features = list(
            features.columns
        )

        if actual_features != expected_features:
            missing_features = [
                feature
                for feature in expected_features
                if feature not in actual_features
            ]

            unexpected_features = [
                feature
                for feature in actual_features
                if feature not in expected_features
            ]

            raise ValueError(
                "Feature schema mismatch. "
                f"Missing features: {missing_features}. "
                f"Unexpected features: {unexpected_features}."
            )