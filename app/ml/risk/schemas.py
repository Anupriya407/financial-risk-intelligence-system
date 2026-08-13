from pydantic import BaseModel, Field, model_validator

from app.ml.risk.categorization import RiskBand

class RiskFactor(BaseModel):
    """A model-derived factor contributing to risk."""

    feature: str = Field(
        min_length=1,
        description="Feature associated with the model prediction.",
    )

    importance: float = Field(
        ge=0.0,
        description="Model-derived feature importance.",
    )


class RiskPrediction(BaseModel):
    """Validated FRIS risk prediction response."""

    risk_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability of financial risk.",
    )

    risk_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Normalized financial risk score.",
    )

    risk_band: RiskBand = Field(
        description="Categorical financial risk band.",
    )

    model_name: str = Field(
        min_length=1,
        description="Name of the ML model used for prediction.",
    )

    model_version: str = Field(
        min_length=1,
        description="Version of the model used for prediction.",
    )

    risk_factors: list[RiskFactor] = Field(
    default_factory=list,
    description="Top model-derived risk factors.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "RiskPrediction":
        """Ensure score and band match the probability."""

        expected_score = round(
            self.risk_probability * 100.0,
            2,
        )

        if self.risk_score != expected_score:
            raise ValueError(
                "Risk score must equal risk probability multiplied by 100."
            )

        if self.risk_probability < 0.30:
            expected_band = RiskBand.LOW
        elif self.risk_probability < 0.60:
            expected_band = RiskBand.MEDIUM
        else:
            expected_band = RiskBand.HIGH

        if self.risk_band != expected_band:
            raise ValueError(
                "Risk band does not match risk probability."
            )

        return self