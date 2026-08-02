"""
POS Cash Balance dataset workflow.
"""

from __future__ import annotations

from app.datasets.workflows.base import BaseDatasetWorkflow


class PosCashWorkflow(BaseDatasetWorkflow):
    """Workflow for the POS_CASH_balance dataset."""

    DATASET_NAME = "pos_cash_balance"


pos_cash_workflow = PosCashWorkflow()