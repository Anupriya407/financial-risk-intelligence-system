from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base_entity import BaseEntity

if TYPE_CHECKING:
    from app.models.company import Company
    from app.models.financial_metrics import FinancialMetrics
    from app.models.risk_assessment import RiskAssessment


class FinancialStatement(BaseEntity):
    """Represents a company's financial statement."""

    __tablename__ = "financial_statements"

    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"),
        nullable=False,
    )

    reporting_period: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    fiscal_year: Mapped[int] = mapped_column(
        nullable=False,
    )

    statement_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    company: Mapped[Company] = relationship(
        back_populates="financial_statements",
    )

    financial_metrics: Mapped[list[FinancialMetrics]] = relationship(
        back_populates="financial_statement",
        cascade="all, delete-orphan",
    )

    risk_assessments: Mapped[list[RiskAssessment]] = relationship(
        back_populates="financial_statement",
    )
