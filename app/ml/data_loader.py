from pathlib import Path

import pandas as pd


class MLDataLoader:
    """Load Phase 15 feature and target artifacts for ML training."""

    def __init__(
        self,
        features_path: Path,
        target_path: Path,
    ) -> None:
        self.features_path = features_path
        self.target_path = target_path

    def load_features(self) -> pd.DataFrame:
        """Load the training feature matrix."""
        return pd.read_parquet(self.features_path)

    def load_target(self) -> pd.DataFrame:
        """Load the training target."""
        return pd.read_parquet(self.target_path)

    def load_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load features and target together."""
        return self.load_features(), self.load_target()