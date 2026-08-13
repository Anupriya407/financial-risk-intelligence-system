from pathlib import Path
from typing import Any

import joblib

from app.ml.risk.categorization import (
    calculate_risk_score,
    categorize_risk,
)
from app.ml.risk.explanation import extract_risk_factors
from app.ml.risk.schemas import RiskPrediction


class RiskPredictor:
    """Generate financial risk predictions."""

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def predict_probability(
        self,
        features: Any,
    ) -> float:
        """Return the probability for exactly one input row."""

        if len(features) != 1:
            raise ValueError(
                "Risk prediction requires exactly one feature row."
            )

        probabilities = self.model.predict_proba(
            features
        )

        return float(
            probabilities[:, 1][0]
        )

    def predict(
        self,
        features: Any,
    ) -> RiskPrediction:
        """Generate a complete validated FRIS risk prediction."""

        probability = self.predict_probability(
            features
        )

        score = calculate_risk_score(
            probability
        )

        band = categorize_risk(
            probability
        )

        risk_factors = extract_risk_factors(
            model=self.model,
            features=features,
            top_n=5,
        )

        return RiskPrediction(
            risk_probability=probability,
            risk_score=score,
            risk_band=band,
            model_name="xgboost",
            model_version="1.0.0",
            risk_factors=risk_factors,
        )