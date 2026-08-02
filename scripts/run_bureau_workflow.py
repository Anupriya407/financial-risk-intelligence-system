"""
Run the Bureau Dataset Workflow.
"""

from app.datasets.workflows.bureau_workflow import bureau_workflow


def main() -> None:
    """Execute the bureau workflow."""

    result = bureau_workflow.run()

    print(result)


if __name__ == "__main__":
    main()