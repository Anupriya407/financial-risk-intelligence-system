"""
Credit Card Balance dataset workflow.
"""

from __future__ import annotations

from app.datasets.workflows.base import BaseDatasetWorkflow


class CreditCardWorkflow(BaseDatasetWorkflow):
    """Workflow for the credit_card_balance dataset."""

    DATASET_NAME = "credit_card_balance"


credit_card_workflow = CreditCardWorkflow()