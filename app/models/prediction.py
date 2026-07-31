from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base_entity import BaseEntity

if TYPE_CHECKING:
    from app.models.risk_assessment import RiskAssessment


class Prediction(BaseEntity):
    """Stores machine learning prediction details."""

    __tablename__ = "predictions"

    risk_assessment_id: Mapped[int] = mapped_column(
        ForeignKey("risk_assessments.id"),
        nullable=False,
    )

    predicted_class: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    prediction_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    risk_assessment: Mapped[RiskAssessment] = relationship(
        back_populates="predictions",
    )
