"""
Dataset profiling for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.loader import dataset_loader


class DatasetProfiler:
    """Generate dataset profiling information."""

    def profile(
        self,
        dataset_name: str,
    ) -> dict[str, object]:
        """Generate a dataset profile."""

        dataframe = dataset_loader.load(dataset_name)

        return {
            "shape": dataframe.shape,
            "columns": dataframe.columns.tolist(),
            "dtypes": dataframe.dtypes.astype(str).to_dict(),
            "missing_values": dataframe.isna().sum().to_dict(),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "memory_usage_bytes": int(
                dataframe.memory_usage(
                    deep=True,
                ).sum(),
            ),
        }


dataset_profiler = DatasetProfiler()