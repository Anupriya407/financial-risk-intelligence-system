"""
Feature engineering orchestrator for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.application_features import (
    application_feature_engineer,
)
from app.feature_engineering.bureau_balance_features import (
    bureau_balance_feature_engineer,
)
from app.feature_engineering.bureau_features import (
    bureau_feature_engineer,
)
from app.feature_engineering.config import (
    APPLICATION_TRAIN_PATH,
    TRAINING_FEATURES_PATH,
    BUREAU_BALANCE_PATH,
    BUREAU_PATH,
    CREDIT_CARD_BALANCE_PATH,
    INSTALLMENTS_PAYMENTS_PATH,
    POS_CASH_BALANCE_PATH,
    PREVIOUS_APPLICATION_PATH,
)
from app.feature_engineering.credit_card_features import (
    credit_card_feature_engineer,
)
from app.feature_engineering.installments_features import (
    installments_feature_engineer,
)
from app.feature_engineering.mergers import (
    feature_merger,
)
from app.feature_engineering.pos_cash_features import (
    pos_cash_feature_engineer,
)
from app.feature_engineering.previous_application_features import (
    previous_application_feature_engineer,
)
from app.feature_engineering.validator import (
    feature_validator,
)


class FeatureEngineeringOrchestrator:
    """Execute the complete FRIS feature engineering workflow."""

    def run(self) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline.

        Returns
        -------
        pandas.DataFrame
            Final application-level feature matrix.
        """

        print("=" * 70)
        print("Financial Risk Intelligence System")
        print("Feature Engineering Workflow")
        print("=" * 70)

        # ---------------------------------------------------------------
        # Load application dataset
        # ---------------------------------------------------------------

        print("\nLoading application dataset...")

        application = pd.read_parquet(
            APPLICATION_TRAIN_PATH,
        )

        print(
            f"Application shape: "
            f"{application.shape}"
        )

        # ---------------------------------------------------------------
        # Application features
        # ---------------------------------------------------------------

        print(
            "\nGenerating application features..."
        )

        application_features = (
            application_feature_engineer.transform(
                application,
            )
        )

        print(
            "Application features generated."
        )

        # ---------------------------------------------------------------
        # Bureau features
        # ---------------------------------------------------------------

        print(
            "\nLoading bureau dataset..."
        )

        bureau = pd.read_parquet(
            BUREAU_PATH,
        )

        print(
            f"Bureau shape: "
            f"{bureau.shape}"
        )

        print(
            "\nGenerating bureau features..."
        )

        bureau_features = (
            bureau_feature_engineer.transform(
                bureau,
            )
        )

        feature_validator.validate(
            bureau_features,
            "SK_ID_CURR",
        )

        print(
            f"Bureau feature shape: "
            f"{bureau_features.shape}"
        )

        # ---------------------------------------------------------------
        # Bureau balance features
        # ---------------------------------------------------------------

        print(
            "\nLoading bureau balance dataset..."
        )

        bureau_balance = pd.read_parquet(
            BUREAU_BALANCE_PATH,
        )

        print(
            f"Bureau balance shape: "
            f"{bureau_balance.shape}"
        )

        print(
            "\nGenerating bureau balance features..."
        )

        bureau_balance_features = (
            bureau_balance_feature_engineer.transform(
                bureau_balance,
                bureau,
            )
        )

        feature_validator.validate(
            bureau_balance_features,
            "SK_ID_CURR",
        )

        print(
            "Bureau balance features generated."
        )

        # ---------------------------------------------------------------
        # Previous application features
        # ---------------------------------------------------------------

        print(
            "\nLoading previous application dataset..."
        )

        previous_application = pd.read_parquet(
            PREVIOUS_APPLICATION_PATH,
        )

        print(
            f"Previous application shape: "
            f"{previous_application.shape}"
        )

        print(
            "\nGenerating previous application features..."
        )

        previous_application_features = (
            previous_application_feature_engineer.transform(
                previous_application,
            )
        )

        feature_validator.validate(
            previous_application_features,
            "SK_ID_CURR",
        )

        # ---------------------------------------------------------------
        # Installment features
        # ---------------------------------------------------------------

        print(
            "\nLoading installments dataset..."
        )

        installments = pd.read_parquet(
            INSTALLMENTS_PAYMENTS_PATH,
        )

        print(
            f"Installments shape: "
            f"{installments.shape}"
        )

        print(
            "\nGenerating installment features..."
        )

        installments_features = (
            installments_feature_engineer.transform(
                installments,
            )
        )

        feature_validator.validate(
            installments_features,
            "SK_ID_CURR",
        )

        # ---------------------------------------------------------------
        # Credit card features
        # ---------------------------------------------------------------

        print(
            "\nLoading credit card dataset..."
        )

        credit_card = pd.read_parquet(
            CREDIT_CARD_BALANCE_PATH,
        )

        print(
            f"Credit card shape: "
            f"{credit_card.shape}"
        )

        print(
            "\nGenerating credit card features..."
        )

        credit_card_features = (
            credit_card_feature_engineer.transform(
                credit_card,
            )
        )

        feature_validator.validate(
            credit_card_features,
            "SK_ID_CURR",
        )

        # ---------------------------------------------------------------
        # POS Cash features
        # ---------------------------------------------------------------

        print(
            "\nLoading POS Cash dataset..."
        )

        pos_cash = pd.read_parquet(
            POS_CASH_BALANCE_PATH,
        )

        print(
            f"POS Cash shape: "
            f"{pos_cash.shape}"
        )

        print(
            "\nGenerating POS Cash features..."
        )

        pos_cash_features = (
            pos_cash_feature_engineer.transform(
                pos_cash,
            )
        )

        feature_validator.validate(
            pos_cash_features,
            "SK_ID_CURR",
        )

        # ---------------------------------------------------------------
        # Merge all feature tables
        # ---------------------------------------------------------------

        print(
            "\nMerging engineered features..."
        )

        result = application_features

        feature_tables = [
            (
                "bureau",
                bureau_features,
            ),
            (
                "bureau_balance",
                bureau_balance_features,
            ),
            (
                "previous_application",
                previous_application_features,
            ),
            (
                "installments",
                installments_features,
            ),
            (
                "credit_card",
                credit_card_features,
            ),
            (
                "pos_cash",
                pos_cash_features,
            ),
        ]

        for name, feature_table in feature_tables:
            print(
                f"Merging {name} features..."
            )

            result = feature_merger.merge_customer_features(
                result,
                feature_table,
            )

            print(
                f"Shape after {name}: "
                f"{result.shape}"
            )

        # ---------------------------------------------------------------
        # Preserve target separately
        # ---------------------------------------------------------------

        print(
            "\nSeparating target from feature matrix..."
        )

        target_path = (
            TRAINING_FEATURES_PATH.parent
            / "fris_training_target.parquet"
        )

        if "TARGET" not in result.columns:
            raise ValueError(
                "TARGET column was not found in the "
                "final engineered DataFrame."
            )

        target = result[
            [
                "SK_ID_CURR",
                "TARGET",
            ]
        ].copy()

        result = result.drop(
            columns=["TARGET"],
        )

        # ---------------------------------------------------------------
        # Final validation
        # ---------------------------------------------------------------

        print(
            "\nValidating final feature matrix..."
        )

        if len(result) != len(application):
            raise ValueError(
                "Final feature matrix changed "
                "the number of application rows."
            )

        feature_validator.validate_no_target_leakage(
            result,
        )

        if "SK_ID_CURR" not in result.columns:
            raise ValueError(
                "SK_ID_CURR is missing from the "
                "final feature matrix."
            )

        if result["SK_ID_CURR"].duplicated().any():
            raise ValueError(
                "Final feature matrix contains "
                "duplicate SK_ID_CURR values."
            )

        if target["SK_ID_CURR"].duplicated().any():
            raise ValueError(
                "Target dataset contains duplicate "
                "SK_ID_CURR values."
            )

        if len(target) != len(application):
            raise ValueError(
                "Target dataset changed "
                "the number of application rows."
            )

        # ---------------------------------------------------------------
        # Save final feature dataset
        # ---------------------------------------------------------------

        TRAINING_FEATURES_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_parquet(
            TRAINING_FEATURES_PATH,
            index=False,
        )

        # ---------------------------------------------------------------
        # Save target dataset
        # ---------------------------------------------------------------

        target.to_parquet(
            target_path,
            index=False,
        )

        # ---------------------------------------------------------------
        # Final output
        # ---------------------------------------------------------------

        print(
            "\nFeature engineering completed successfully."
        )

        print(
            f"Final rows    : "
            f"{len(result):,}"
        )

        print(
            f"Final columns : "
            f"{len(result.columns):,}"
        )

        print(
            f"Features      : "
            f"{TRAINING_FEATURES_PATH}"
        )

        print(
            f"Target        : "
            f"{target_path}"
        )

        return result


feature_engineering_orchestrator = (
    FeatureEngineeringOrchestrator()
)