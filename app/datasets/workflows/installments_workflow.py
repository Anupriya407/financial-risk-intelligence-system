"""
Installments Payments dataset workflow.
"""

from __future__ import annotations

from app.datasets.workflows.base import BaseDatasetWorkflow


class InstallmentsWorkflow(BaseDatasetWorkflow):
    """Workflow for the installments_payments dataset."""

    DATASET_NAME = "installments_payments"


installments_workflow = InstallmentsWorkflow()