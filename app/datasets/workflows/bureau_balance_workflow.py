"""
Bureau Balance dataset workflow.
"""

from __future__ import annotations

from app.datasets.workflows.base import BaseDatasetWorkflow


class BureauBalanceWorkflow(BaseDatasetWorkflow):
    """Workflow for the bureau_balance dataset."""

    DATASET_NAME = "bureau_balance"


bureau_balance_workflow = BureauBalanceWorkflow()