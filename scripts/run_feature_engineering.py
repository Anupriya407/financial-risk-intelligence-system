"""
Run the complete FRIS feature engineering workflow.
"""

from __future__ import annotations

from app.feature_engineering.orchestrator import (
    feature_engineering_orchestrator,
)


def main() -> None:
    """Run the feature engineering workflow."""

    result = feature_engineering_orchestrator.run()

    print("\n" + "=" * 70)
    print("FINAL FEATURE DATASET")
    print("=" * 70)

    print(
        f"Rows    : {len(result):,}"
    )

    print(
        f"Columns : {len(result.columns):,}"
    )


if __name__ == "__main__":
    main()