from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base_entity import BaseEntity

if TYPE_CHECKING:
    from app.models.financial_statement import FinancialStatement


class FinancialMetrics(BaseEntity):
    """Stores calculated financial metrics for a financial statement."""

    __tablename__ = "financial_metrics"

    financial_statement_id: Mapped[int] = mapped_column(
        ForeignKey("financial_statements.id"),
        nullable=False,
    )

    current_ratio: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    debt_to_equity_ratio: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    return_on_assets: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    return_on_equity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    net_profit_margin: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    financial_statement: Mapped[FinancialStatement] = relationship(
        back_populates="financial_metrics",
    )
