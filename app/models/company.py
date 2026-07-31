from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base_entity import BaseEntity

if TYPE_CHECKING:
    from app.models.financial_statement import FinancialStatement
    from app.models.risk_assessment import RiskAssessment


class Company(BaseEntity):
    """Represents a company in the Financial Risk Intelligence System."""

    __tablename__ = "companies"

    company_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    ticker_symbol: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False,
    )

    industry: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    country: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    financial_statements: Mapped[list[FinancialStatement]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )

    risk_assessments: Mapped[list[RiskAssessment]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )
