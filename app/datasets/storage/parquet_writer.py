"""
Parquet writer for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetWriter:
    """Write pandas DataFrames to Parquet files."""

    def write(
        self,
        dataframe: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """
        Write a DataFrame to a Parquet file.

        Parameters
        ----------
        dataframe:
            DataFrame to save.

        output_path:
            Destination Parquet file.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_parquet(
            output_path,
            index=False,
        )


parquet_writer = ParquetWriter()