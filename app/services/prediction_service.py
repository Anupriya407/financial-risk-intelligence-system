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

        return self.prediction_repository.create(prediction)

    def get_prediction_by_id(
        self,
        prediction_id: int,
    ) -> Prediction | None:
        """Retrieve a prediction by its ID."""

        return self.prediction_repository.get_by_id(prediction_id)

    def get_all_predictions(
        self,
    ) -> list[Prediction]:
        """Retrieve all predictions."""

        return self.prediction_repository.get_all()

    def get_predictions_by_risk_assessment(
        self,
        risk_assessment_id: int,
    ) -> list[Prediction]:
        """Retrieve predictions for a risk assessment."""

        return self.prediction_repository.get_by_risk_assessment(risk_assessment_id)

    def get_latest_prediction(
        self,
        risk_assessment_id: int,
    ) -> Prediction | None:
        """Retrieve the latest prediction for a risk assessment."""

        return self.prediction_repository.get_latest_prediction(risk_assessment_id)

    def get_predictions_by_predicted_class(
        self,
        predicted_class: str,
    ) -> list[Prediction]:
        """Retrieve predictions by predicted class."""

        return self.prediction_repository.get_by_predicted_class(predicted_class)

    def get_predictions_by_model_name(
        self,
        model_name: str,
    ) -> list[Prediction]:
        """Retrieve predictions by model name."""

        return self.prediction_repository.get_by_model_name(model_name)

    def get_predictions_by_model_version(
        self,
        model_version: str,
    ) -> list[Prediction]:
        """Retrieve predictions by model version."""

        return self.prediction_repository.get_by_model_version(model_version)

    def update_prediction(
        self,
        prediction_id: int,
        prediction_data: Prediction,
    ) -> Prediction:
        """Update a prediction."""

        prediction = self.prediction_repository.get_by_id(prediction_id)

        if prediction is None:
            raise ValueError("Prediction not found.")

        prediction.risk_assessment_id = prediction_data.risk_assessment_id
        prediction.predicted_class = prediction_data.predicted_class
        prediction.confidence_score = prediction_data.confidence_score
        prediction.model_name = prediction_data.model_name
        prediction.model_version = prediction_data.model_version
        prediction.prediction_metadata = prediction_data.prediction_metadata

        return self.prediction_repository.update(prediction)

    def delete_prediction(
        self,
        prediction_id: int,
    ) -> None:
        """Delete a prediction."""

        prediction = self.prediction_repository.get_by_id(prediction_id)

        if prediction is None:
            raise ValueError("Prediction not found.")

        self.prediction_repository.delete(prediction)
