"""
Application dataset workflow.
"""

from __future__ import annotations

from app.datasets.workflows.base import BaseDatasetWorkflow


class ApplicationWorkflow(BaseDatasetWorkflow):
    """Workflow for application_train."""

    DATASET_NAME = "application_train"


application_workflow = ApplicationWorkflow()