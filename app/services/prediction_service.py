from app.models.prediction import Prediction
from app.repositories.prediction_repository import (
    PredictionRepository,
)


class PredictionService:
    """Service layer for Prediction business logic."""

    def __init__(
        self,
        prediction_repository: PredictionRepository,
    ) -> None:
        self.prediction_repository = prediction_repository

    def create_prediction(
        self,
        prediction: Prediction,
    ) -> Prediction:
        """Create a prediction."""

        return self.prediction_repository.create(
            prediction
        )

    def get_prediction_by_id(
        self,
        prediction_id: int,
    ) -> Prediction | None:
        """Retrieve a prediction by its ID."""

        return self.prediction_repository.get_by_id(
            prediction_id
        )

    def get_predictions_by_company(
        self,
        company_id: int,
    ) -> list[Prediction]:
        """Retrieve all predictions for a company."""

        return self.prediction_repository.get_by_company(
            company_id
        )

    def get_latest_prediction(
        self,
        company_id: int,
    ) -> Prediction | None:
        """Retrieve the latest prediction for a company."""

        return self.prediction_repository.get_latest_prediction(
            company_id
        )

    def get_predictions_by_label(
        self,
        prediction_label: str,
    ) -> list[Prediction]:
        """Retrieve predictions by prediction label."""

        return (
            self.prediction_repository.get_by_prediction_label(
                prediction_label
            )
        )

    def get_predictions_by_model_version(
        self,
        model_version: str,
    ) -> list[Prediction]:
        """Retrieve predictions by model version."""

        return (
            self.prediction_repository.get_by_model_version(
                model_version
            )
        )

    def update_prediction(
        self,
        prediction: Prediction,
    ) -> Prediction:
        """Update a prediction."""

        existing_prediction = (
            self.prediction_repository.get_by_id(
                prediction.id
            )
        )

        if existing_prediction is None:
            raise ValueError(
                "Prediction not found."
            )

        return self.prediction_repository.update(
            prediction
        )

    def delete_prediction(
        self,
        prediction_id: int,
    ) -> None:
        """Delete a prediction."""

        prediction = (
            self.prediction_repository.get_by_id(
                prediction_id
            )
        )

        if prediction is None:
            raise ValueError(
                "Prediction not found."
            )

        self.prediction_repository.delete(
            prediction
        )