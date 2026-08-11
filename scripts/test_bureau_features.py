"""
Manual verification of bureau feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.bureau_features import (
    bureau_feature_engineer,
)
from app.feature_engineering.config import BUREAU_PATH


def main() -> None:
    """Run bureau feature engineering verification."""

    print("=" * 70)
    print("Bureau Feature Engineering Verification")
    print("=" * 70)

    dataframe = pd.read_parquet(
        BUREAU_PATH,
    )

    print(
        f"Original rows    : {len(dataframe):,}"
    )

    print(
        f"Original columns : "
        f"{len(dataframe.columns):,}"
    )

    result = bureau_feature_engineer.transform(
        dataframe,
    )

    generated_features = [
        column
        for column in result.columns
        if column.startswith("FE_BUREAU_")
    ]

    print(
        f"\nCustomer rows    : "
        f"{len(result):,}"
    )

    print(
        f"Generated features: "
        f"{len(generated_features):,}"
    )

    print("\nGenerated features:")

    for feature in generated_features:
        print(f"  - {feature}")

    print("\nDuplicate customer keys:")

    print(
        result["SK_ID_CURR"]
        .duplicated()
        .sum()
    )

    print("\nVerification completed successfully.")


if __name__ == "__main__":
    main()