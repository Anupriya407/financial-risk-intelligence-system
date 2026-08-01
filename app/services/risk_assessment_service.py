from app.models.risk_assessment import RiskAssessment
from app.repositories.risk_assessment_repository import (
    RiskAssessmentRepository,
)


class RiskAssessmentService:
    """Service layer for RiskAssessment business logic."""

    def __init__(
        self,
        risk_assessment_repository: RiskAssessmentRepository,
    ) -> None:
        self.risk_assessment_repository = risk_assessment_repository

    def create_risk_assessment(
        self,
        risk_assessment: RiskAssessment,
    ) -> RiskAssessment:
        """Create a risk assessment."""

        return self.risk_assessment_repository.create(risk_assessment)

    def get_risk_assessment_by_id(
        self,
        risk_assessment_id: int,
    ) -> RiskAssessment | None:
        """Retrieve a risk assessment by ID."""

        return self.risk_assessment_repository.get_by_id(risk_assessment_id)

    def get_all_risk_assessments(
        self,
    ) -> list[RiskAssessment]:
        """Retrieve all risk assessments."""

        return self.risk_assessment_repository.get_all()

    def get_risk_assessments_by_company(
        self,
        company_id: int,
    ) -> list[RiskAssessment]:
        """Retrieve all risk assessments for a company."""

        return self.risk_assessment_repository.get_by_company(company_id)

    def get_risk_assessments_by_risk_level(
        self,
        risk_level: str,
    ) -> list[RiskAssessment]:
        """Retrieve risk assessments by risk level."""

        return self.risk_assessment_repository.get_by_risk_level(risk_level)

    def get_latest_risk_assessment(
        self,
        company_id: int,
    ) -> RiskAssessment | None:
        """Retrieve the latest risk assessment for a company."""

        return self.risk_assessment_repository.get_latest_assessment(company_id)

    def get_high_risk_companies(
        self,
    ) -> list[RiskAssessment]:
        """Retrieve all high-risk company assessments."""

        return self.risk_assessment_repository.get_high_risk_companies()

    def update_risk_assessment(
        self,
        risk_assessment_id: int,
        risk_assessment_data: RiskAssessment,
    ) -> RiskAssessment:
        """Update a risk assessment."""

        risk_assessment = self.risk_assessment_repository.get_by_id(risk_assessment_id)

        if risk_assessment is None:
            raise ValueError("Risk assessment not found.")

        risk_assessment.company_id = risk_assessment_data.company_id
        risk_assessment.financial_statement_id = risk_assessment_data.financial_statement_id
        risk_assessment.risk_score = risk_assessment_data.risk_score
        risk_assessment.risk_level = risk_assessment_data.risk_level
        risk_assessment.model_version = risk_assessment_data.model_version

        return self.risk_assessment_repository.update(risk_assessment)

    def delete_risk_assessment(
        self,
        risk_assessment_id: int,
    ) -> None:
        """Delete a risk assessment."""

        risk_assessment = self.risk_assessment_repository.get_by_id(risk_assessment_id)

        if risk_assessment is None:
            raise ValueError("Risk assessment not found.")

        self.risk_assessment_repository.delete(risk_assessment)
