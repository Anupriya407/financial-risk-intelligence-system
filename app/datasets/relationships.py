"""
Dataset relationship validation for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.loader import dataset_loader


class DatasetRelationshipValidator:
    """Validate relationships between datasets."""

    def validate_foreign_key(
        self,
        parent_dataset: str,
        parent_key: str,
        child_dataset: str,
        child_key: str,
    ) -> None:
        """Validate that every foreign key exists in the parent dataset."""

        parent = dataset_loader.load(
            parent_dataset,
            usecols=[parent_key],
        )

        child = dataset_loader.load(
            child_dataset,
            usecols=[child_key],
        )

        parent_keys = set(parent[parent_key].dropna().unique())
        child_keys = set(child[child_key].dropna().unique())

        missing = child_keys - parent_keys

        if missing:
            raise ValueError(
                f"{len(missing)} missing foreign keys "
                f"from '{child_dataset}.{child_key}' "
                f"referencing "
                f"'{parent_dataset}.{parent_key}'."
            )

    def validate_home_credit_relationships(
        self,
    ) -> None:
        """
        Validate Home Credit dataset relationships.

        Notes
        -----
        The Home Credit auxiliary datasets contain records for both
        application_train.csv and application_test.csv.

        Therefore, we do not validate SK_ID_CURR relationships
        against application_train.csv.
        """

        self.validate_foreign_key(
            parent_dataset="bureau",
            parent_key="SK_ID_BUREAU",
            child_dataset="bureau_balance",
            child_key="SK_ID_BUREAU",
        )

        self.validate_foreign_key(
            parent_dataset="previous_application",
            parent_key="SK_ID_PREV",
            child_dataset="installments_payments",
            child_key="SK_ID_PREV",
        )

        self.validate_foreign_key(
            parent_dataset="previous_application",
            parent_key="SK_ID_PREV",
            child_dataset="credit_card_balance",
            child_key="SK_ID_PREV",
        )

        self.validate_foreign_key(
            parent_dataset="previous_application",
            parent_key="SK_ID_PREV",
            child_dataset="pos_cash_balance",
            child_key="SK_ID_PREV",
        )


dataset_relationship_validator = DatasetRelationshipValidator()