"""
Dataset cache for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd


class DatasetCache:
    """In-memory cache for loaded datasets."""

    def __init__(self) -> None:
        """Initialize the dataset cache."""

        self._cache: dict[str, pd.DataFrame] = {}

    def has(
        self,
        dataset_name: str,
    ) -> bool:
        """Return True if the dataset is cached."""

        return dataset_name in self._cache

    def get(
        self,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Return a copy of a cached dataset."""

        return self._cache[dataset_name].copy(
            deep=True,
        )

    def set(
        self,
        dataset_name: str,
        dataframe: pd.DataFrame,
    ) -> None:
        """Cache a copy of a dataset."""

        self._cache[dataset_name] = dataframe.copy(
            deep=True,
        )

    def clear(self) -> None:
        """Clear the cache."""

        self._cache.clear()

    def clear_dataset(
        self,
        dataset_name: str,
    ) -> None:
        """Remove a dataset from the cache."""

        self._cache.pop(
            dataset_name,
            None,
        )


dataset_cache = DatasetCache()