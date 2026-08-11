"""
Configuration for the Financial Risk Intelligence System
feature engineering pipeline.
"""

from __future__ import annotations

from app.datasets.config import (
    DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Feature Engineering Directories
# ---------------------------------------------------------------------------

FEATURE_ENGINEERING_DATA_DIR = DATA_DIR / "feature_engineering"

FEATURE_METADATA_DIR = (
    FEATURE_ENGINEERING_DATA_DIR / "metadata"
)

FEATURE_REPORTS_DIR = (
    FEATURE_ENGINEERING_DATA_DIR / "reports"
)


# ---------------------------------------------------------------------------
# Input Datasets
# ---------------------------------------------------------------------------

APPLICATION_TRAIN_PATH = (
    INTERIM_DATA_DIR / "application_train.parquet"
)

BUREAU_PATH = (
    INTERIM_DATA_DIR / "bureau.parquet"
)

BUREAU_BALANCE_PATH = (
    INTERIM_DATA_DIR / "bureau_balance.parquet"
)

PREVIOUS_APPLICATION_PATH = (
    INTERIM_DATA_DIR / "previous_application.parquet"
)

INSTALLMENTS_PAYMENTS_PATH = (
    INTERIM_DATA_DIR / "installments_payments.parquet"
)

CREDIT_CARD_BALANCE_PATH = (
    INTERIM_DATA_DIR / "credit_card_balance.parquet"
)

POS_CASH_BALANCE_PATH = (
    INTERIM_DATA_DIR / "pos_cash_balance.parquet"
)


# ---------------------------------------------------------------------------
# Output Datasets
# ---------------------------------------------------------------------------

FEATURE_DATA_DIR = PROCESSED_DATA_DIR / "features"

TRAINING_FEATURES_PATH = (
    FEATURE_DATA_DIR / "fris_training_features.parquet"
)


# ---------------------------------------------------------------------------
# Primary Keys
# ---------------------------------------------------------------------------

APPLICATION_KEY = "SK_ID_CURR"

BUREAU_KEY = "SK_ID_BUREAU"

PREVIOUS_APPLICATION_KEY = "SK_ID_PREV"


# ---------------------------------------------------------------------------
# Dataset Names
# ---------------------------------------------------------------------------

APPLICATION_DATASET = "application_train"

BUREAU_DATASET = "bureau"

BUREAU_BALANCE_DATASET = "bureau_balance"

PREVIOUS_APPLICATION_DATASET = "previous_application"

INSTALLMENTS_DATASET = "installments_payments"

CREDIT_CARD_DATASET = "credit_card_balance"

POS_CASH_DATASET = "pos_cash_balance"


# ---------------------------------------------------------------------------
# Feature Engineering Settings
# ---------------------------------------------------------------------------

TARGET_COLUMN = "TARGET"

DEFAULT_AGGREGATION_PREFIX = "FE"


__all__ = [
    "FEATURE_ENGINEERING_DATA_DIR",
    "FEATURE_METADATA_DIR",
    "FEATURE_REPORTS_DIR",
    "APPLICATION_TRAIN_PATH",
    "BUREAU_PATH",
    "BUREAU_BALANCE_PATH",
    "PREVIOUS_APPLICATION_PATH",
    "INSTALLMENTS_PAYMENTS_PATH",
    "CREDIT_CARD_BALANCE_PATH",
    "POS_CASH_BALANCE_PATH",
    "FEATURE_DATA_DIR",
    "TRAINING_FEATURES_PATH",
    "APPLICATION_KEY",
    "BUREAU_KEY",
    "PREVIOUS_APPLICATION_KEY",
    "APPLICATION_DATASET",
    "BUREAU_DATASET",
    "BUREAU_BALANCE_DATASET",
    "PREVIOUS_APPLICATION_DATASET",
    "INSTALLMENTS_DATASET",
    "CREDIT_CARD_DATASET",
    "POS_CASH_DATASET",
    "TARGET_COLUMN",
    "DEFAULT_AGGREGATION_PREFIX",
]