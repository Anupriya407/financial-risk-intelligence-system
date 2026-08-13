from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class HoldoutSplit:
    """Development and final test datasets."""

    X_development: pd.DataFrame
    X_test: pd.DataFrame
    y_development: pd.Series
    y_test: pd.Series


def create_holdout_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.15,
    random_state: int = 42,
) -> HoldoutSplit:
    """Create a stratified development/final-test split."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if len(features) != len(target):
        raise ValueError(
            "Features and target must contain the same number of rows."
        )

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return HoldoutSplit(
        X_development=X_development,
        X_test=X_test,
        y_development=y_development,
        y_test=y_test,
    )