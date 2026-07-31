from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base_entity import BaseEntity

if TYPE_CHECKING:
    from app.models.company import Company
    from app.models.financial_statement import FinancialStatement
    from app.models.prediction import Prediction


class RiskAssessment(BaseEntity):
    """Represents the financial risk assessment of a company."""

    __tablename__ = "risk_assessments"

    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"),
        nullable=False,
    )

    financial_statement_id: Mapped[int] = mapped_column(
        ForeignKey("financial_statements.id"),
        nullable=False,
    )

    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    risk_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    company: Mapped[Company] = relationship(
        back_populates="risk_assessments",
    )

    financial_statement: Mapped[FinancialStatement] = relationship(
        back_populates="risk_assessments",
    )

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="risk_assessment",
        cascade="all, delete-orphan",
    )
