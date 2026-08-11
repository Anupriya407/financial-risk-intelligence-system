"""
Manual verification of application feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.application_features import (
    application_feature_engineer,
)
from app.feature_engineering.config import APPLICATION_TRAIN_PATH


def main() -> None:
    """Run application feature engineering verification."""

    print("=" * 70)
    print("Application Feature Engineering Verification")
    print("=" * 70)

    dataframe = pd.read_parquet(
        APPLICATION_TRAIN_PATH,
    )

    print(f"Original rows    : {len(dataframe):,}")
    print(f"Original columns : {len(dataframe.columns):,}")

    result = application_feature_engineer.transform(
        dataframe,
    )

    generated_features = [
        column
        for column in result.columns
        if column.startswith("FE_")
    ]

    print(
        f"Generated features: "
        f"{len(generated_features):,}"
    )

    print("\nGenerated features:")

    for feature in generated_features:
        print(f"  - {feature}")

    print("\nFinal shape:")
    print(
        f"Rows    : {result.shape[0]:,}"
    )
    print(
        f"Columns : {result.shape[1]:,}"
    )

    print("\nVerification completed successfully.")


if __name__ == "__main__":
    main()