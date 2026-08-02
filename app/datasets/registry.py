"""
Dataset registry for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from app.datasets.config import RAW_DATA_DIR
from app.datasets.metadata import DatasetMetadata


class DatasetRegistry:
    """Central registry for all datasets used in FRIS."""

    def __init__(self) -> None:
        """Initialize the dataset registry."""

        self._datasets: dict[str, DatasetMetadata] = {}

        self._register_home_credit_datasets()

    def _register_home_credit_datasets(self) -> None:
        """Register Home Credit datasets."""

        home_credit_dir = RAW_DATA_DIR / "home_credit"

        datasets = [
            DatasetMetadata(
                name="application_train",
                filename="application_train.csv",
                description="Training loan application dataset.",
                primary_key="SK_ID_CURR",
                path=home_credit_dir / "application_train.csv",
            ),
            DatasetMetadata(
                name="application_test",
                filename="application_test.csv",
                description="Inference loan application dataset.",
                primary_key="SK_ID_CURR",
                path=home_credit_dir / "application_test.csv",
            ),
            DatasetMetadata(
                name="bureau",
                filename="bureau.csv",
                description="External credit bureau history.",
                primary_key="SK_ID_BUREAU",
                path=home_credit_dir / "bureau.csv",
            ),
            DatasetMetadata(
                name="bureau_balance",
                filename="bureau_balance.csv",
                description="Monthly bureau credit history.",
                primary_key=("SK_ID_BUREAU", "MONTHS_BALANCE"),
                path=home_credit_dir / "bureau_balance.csv",
            ),
            DatasetMetadata(
                name="previous_application",
                filename="previous_application.csv",
                description="Previous Home Credit applications.",
                primary_key="SK_ID_PREV",
                path=home_credit_dir / "previous_application.csv",
            ),
            DatasetMetadata(
                name="installments_payments",
                filename="installments_payments.csv",
                description="Historical installment payments.",
                primary_key=("SK_ID_PREV", "NUM_INSTALMENT_NUMBER"),
                path=home_credit_dir / "installments_payments.csv",
            ),
            DatasetMetadata(
                name="credit_card_balance",
                filename="credit_card_balance.csv",
                description="Monthly credit card history.",
                primary_key=("SK_ID_PREV", "MONTHS_BALANCE"),
                path=home_credit_dir / "credit_card_balance.csv",
            ),
            DatasetMetadata(
                name="pos_cash_balance",
                filename="POS_CASH_balance.csv",
                description="POS and cash loan history.",
                primary_key=("SK_ID_PREV", "MONTHS_BALANCE"),
                path=home_credit_dir / "POS_CASH_balance.csv",
            ),
            DatasetMetadata(
                name="column_description",
                filename="HomeCredit_columns_description.csv",
                description="Official Home Credit column descriptions.",
                primary_key="Row",
                path=home_credit_dir / "HomeCredit_columns_description.csv",
            ),
        ]

        for dataset in datasets:
            self._datasets[dataset.name] = dataset

    def get_dataset(self, name: str) -> DatasetMetadata:
        """
        Retrieve dataset metadata by name.

        Raises:
            KeyError:
                If the dataset is not registered.
        """
        try:
            return self._datasets[name]
        except KeyError as exc:
            raise KeyError(
                f"Dataset '{name}' is not registered.",
            ) from exc

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return metadata for all registered datasets."""
        return list(self._datasets.values())

    def list_dataset_names(self) -> list[str]:
        """Return the names of all registered datasets."""
        return sorted(self._datasets.keys())

    def dataset_exists(self, name: str) -> bool:
        """Return True if the registered dataset exists on disk."""
        return self.get_dataset(name).exists

    def validate_registry(self) -> None:
        """
        Validate that all registered datasets exist.

        Raises:
            FileNotFoundError:
                If one or more datasets are missing.
        """
        missing = [
            dataset.path
            for dataset in self._datasets.values()
            if not dataset.exists
        ]

        if missing:
            missing_files = "\n".join(
                f"- {path}"
                for path in missing
            )

            raise FileNotFoundError(
                "The following datasets were not found:\n"
                f"{missing_files}",
            )


dataset_registry = DatasetRegistry()