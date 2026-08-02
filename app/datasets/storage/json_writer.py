"""
JSON writer for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonWriter:
    """Write Python objects to JSON files."""

    def write(
        self,
        data: dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Write a dictionary to a JSON file.

        Parameters
        ----------
        data:
            Data to serialize.

        output_path:
            Destination JSON file.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )


json_writer = JsonWriter()