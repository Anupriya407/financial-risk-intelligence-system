"""
Previous Application dataset workflow.
"""

from __future__ import annotations

from app.datasets.workflows.base import BaseDatasetWorkflow


class PreviousApplicationWorkflow(BaseDatasetWorkflow):
    """Workflow for the previous_application dataset."""

    DATASET_NAME = "previous_application"


previous_application_workflow = PreviousApplicationWorkflow()