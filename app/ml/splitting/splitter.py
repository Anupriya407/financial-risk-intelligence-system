from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train, validation, and test datasets."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class DatasetSplitter:
    """Create reproducible stratified train/validation/test splits."""

    def __init__(
        self,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        random_state: int = 42,
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")

        if test_size + validation_size >= 1:
            raise ValueError(
                "test_size + validation_size must be less than 1."
            )

        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def split(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> DatasetSplit:
        """Create stratified train, validation, and test splits."""

        if len(features) != len(target):
            raise ValueError(
                "Features and target must contain the same number of rows."
            )

        X_train_validation, X_test, y_train_validation, y_test = (
            train_test_split(
                features,
                target,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=target,
            )
        )

        validation_ratio = self.validation_size / (
            1 - self.test_size
        )

        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=validation_ratio,
            random_state=self.random_state,
            stratify=y_train_validation,
        )

        return DatasetSplit(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )