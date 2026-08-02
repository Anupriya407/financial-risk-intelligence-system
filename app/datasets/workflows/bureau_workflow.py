"""
Bureau dataset workflow.
"""

from __future__ import annotations

from app.datasets.workflows.base import BaseDatasetWorkflow


class BureauWorkflow(BaseDatasetWorkflow):
    """Workflow for bureau."""

    DATASET_NAME = "bureau"


bureau_workflow = BureauWorkflow()