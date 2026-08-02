"""
Run the Application Dataset Workflow.
"""

from __future__ import annotations

from app.datasets.workflows.application_workflow import (
    application_workflow,
)


def main() -> None:
    """Execute the application dataset workflow."""

    print("=" * 60)
    print("Financial Risk Intelligence System")
    print("Application Dataset Workflow")
    print("=" * 60)

    result = application_workflow.run()

    print("\nWorkflow completed successfully.\n")

    print(f"Dataset             : {result.dataset_name}")
    print(f"Rows                : {result.rows}")
    print(f"Columns             : {result.columns}")
    print(f"Parquet             : {result.parquet_path}")
    print(f"Profile Report      : {result.profile_path}")
    print(f"Quality Report      : {result.quality_report_path}")
    print(
        "Preprocessing Report: "
        f"{result.preprocessing_report_path}"
    )


if __name__ == "__main__":
    main()