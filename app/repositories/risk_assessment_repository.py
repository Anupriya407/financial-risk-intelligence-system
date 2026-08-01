from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models.risk_assessment import RiskAssessment
from app.repositories.base import BaseRepository


class RiskAssessmentRepository(BaseRepository[RiskAssessment]):
    """Repository for RiskAssessment database operations."""

    def __init__(
        self,
        db: Session,
    ) -> None:
        super().__init__(
            db=db,
            model=RiskAssessment,
        )

    def get_by_company(
        self,
        company_id: int,
    ) -> list[RiskAssessment]:
        """Retrieve all risk assessments for a company."""

        statement = select(RiskAssessment).where(RiskAssessment.company_id == company_id)

        return list(self.db.scalars(statement).all())

    def get_by_risk_level(
        self,
        risk_level: str,
    ) -> list[RiskAssessment]:
        """Retrieve all risk assessments with a specific risk level."""

        statement = select(RiskAssessment).where(RiskAssessment.risk_level == risk_level)

        return list(self.db.scalars(statement).all())

    def get_latest_assessment(
        self,
        company_id: int,
    ) -> RiskAssessment | None:
        """Retrieve the latest risk assessment for a company."""

        statement = (
            select(RiskAssessment)
            .where(RiskAssessment.company_id == company_id)
            .order_by(desc(RiskAssessment.assessment_date))
        )

        return self.db.scalars(statement).first()

    def get_high_risk_companies(
        self,
    ) -> list[RiskAssessment]:
        """Retrieve all high-risk assessments."""

        statement = select(RiskAssessment).where(RiskAssessment.risk_level == "High")

        return list(self.db.scalars(statement).all())
