from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_numeric_features(X: Any):
    """Select numeric feature columns."""
    return X.select_dtypes(include=["number"]).columns


def select_categorical_features(X: Any):
    """Select categorical feature columns."""
    return X.select_dtypes(
        include=["object", "category", "string"]
    ).columns


class MLPreprocessor:
    """Build and manage the ML preprocessing pipeline."""

    def __init__(self) -> None:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    select_numeric_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    select_categorical_features,
                ),
            ],
            remainder="drop",
        )

    def fit(self, X):
        """Fit preprocessing on training data only."""
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        """Transform data using the fitted preprocessing pipeline."""
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        """Fit and transform training data."""
        return self.pipeline.fit_transform(X)