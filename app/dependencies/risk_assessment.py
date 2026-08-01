from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.risk_assessment_repository import (
    RiskAssessmentRepository,
)
from app.services.risk_assessment_service import (
    RiskAssessmentService,
)


def get_risk_assessment_repository(
    db: Session = Depends(get_db),
) -> RiskAssessmentRepository:
    """Return a RiskAssessmentRepository instance."""

    return RiskAssessmentRepository(db)


def get_risk_assessment_service(
    risk_assessment_repository: RiskAssessmentRepository = Depends(
        get_risk_assessment_repository,
    ),
) -> RiskAssessmentService:
    """Return a RiskAssessmentService instance."""

    return RiskAssessmentService(
        risk_assessment_repository,
    )
