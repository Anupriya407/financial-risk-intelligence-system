"""
Run all dataset workflows.
"""

from app.datasets.workflows.orchestrator import (
    dataset_workflow_orchestrator,
)


def main() -> None:
    """Run all workflows."""

    results = dataset_workflow_orchestrator.run()

    print("\n")
    print("=" * 70)
    print("Dataset Engineering Completed")
    print("=" * 70)

    for result in results:
        print(
            f"{result.dataset_name:<25}"
            f"{result.rows:,} rows"
        )


if __name__ == "__main__":
    main()