"""
Dataset loader for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.cache import dataset_cache
from app.datasets.registry import dataset_registry


class DatasetLoader:
    """Loads datasets registered in the dataset registry."""

    def load(
        self,
        dataset_name: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load a dataset into a pandas DataFrame.

        Parameters
        ----------
        dataset_name:
            Registered dataset name.

        **kwargs:
            Additional keyword arguments passed to
            pandas.read_csv().

        Returns
        -------
        pandas.DataFrame
            Loaded pandas DataFrame.
        """

        # Use the cache only when loading the
        # complete dataset (no custom read_csv options).
        if not kwargs and dataset_cache.has(dataset_name):
            print(f"Cache hit: {dataset_name}")

            return dataset_cache.get(
                dataset_name,
            )

        dataset = dataset_registry.get_dataset(
            dataset_name,
        )

        if not dataset.exists:
            raise FileNotFoundError(
                f"Dataset '{dataset.filename}' not found "
                f"at '{dataset.path}'.",
            )

        print(f"Loading dataset: {dataset.filename}")

        encodings = (
            "utf-8",
            "utf-8-sig",
            "cp1252",
            "latin1",
        )

        last_error: UnicodeDecodeError | None = None

        for encoding in encodings:
            try:
                dataframe = pd.read_csv(
                    dataset.path,
                    encoding=encoding,
                    **kwargs,
                )

                # Cache only complete datasets.
                if not kwargs:
                    dataset_cache.set(
                        dataset_name,
                        dataframe,
                    )

                return dataframe

            except UnicodeDecodeError as exc:
                last_error = exc

        raise RuntimeError(
            f"Unable to read dataset '{dataset.filename}' "
            "using the supported encodings.",
        ) from last_error


dataset_loader = DatasetLoader()