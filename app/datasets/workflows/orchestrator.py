"""
Dataset workflow orchestrator for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from app.datasets.workflows.application_workflow import (
    application_workflow,
)
from app.datasets.workflows.bureau_workflow import (
    bureau_workflow,
)
from app.datasets.workflows.bureau_balance_workflow import (
    bureau_balance_workflow,
)
from app.datasets.workflows.previous_application_workflow import (
    previous_application_workflow,
)
from app.datasets.workflows.installments_workflow import (
    installments_workflow,
)
from app.datasets.workflows.credit_card_workflow import (
    credit_card_workflow,
)
from app.datasets.workflows.pos_cash_workflow import (
    pos_cash_workflow,
)


class DatasetWorkflowOrchestrator:
    """Execute all dataset workflows."""

    def __init__(self) -> None:
        """Initialize workflow collection."""

        self.workflows = [
            application_workflow,
            bureau_workflow,
            bureau_balance_workflow,
            previous_application_workflow,
            installments_workflow,
            credit_card_workflow,
            pos_cash_workflow,
        ]

    def run(self) -> list:
        """
        Execute every dataset workflow.

        Returns
        -------
        list
            Workflow results.
        """

        results = []

        for workflow in self.workflows:
            print("=" * 70)
            print(f"Running {workflow.DATASET_NAME}")
            print("=" * 70)

            results.append(
                workflow.run(),
            )

        return results


dataset_workflow_orchestrator = (
    DatasetWorkflowOrchestrator()
)