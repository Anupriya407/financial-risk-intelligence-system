from pathlib import Path

import pandas as pd

from app.ml.data_loader import MLDataLoader


def test_load_features(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"

    expected = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )
    expected.to_parquet(features_path)

    loader = MLDataLoader(
        features_path=features_path,
        target_path=tmp_path / "target.parquet",
    )

    result = loader.load_features()

    pd.testing.assert_frame_equal(result, expected)


def test_load_target(tmp_path: Path) -> None:
    target_path = tmp_path / "target.parquet"

    expected = pd.DataFrame({"TARGET": [0, 1]})
    expected.to_parquet(target_path)

    loader = MLDataLoader(
        features_path=tmp_path / "features.parquet",
        target_path=target_path,
    )

    result = loader.load_target()

    pd.testing.assert_frame_equal(result, expected)


def test_load_training_data(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    target_path = tmp_path / "target.parquet"

    features = pd.DataFrame({"feature_a": [1.0, 2.0]})
    target = pd.DataFrame({"TARGET": [0, 1]})

    features.to_parquet(features_path)
    target.to_parquet(target_path)

    loader = MLDataLoader(
        features_path=features_path,
        target_path=target_path,
    )

    loaded_features, loaded_target = loader.load_training_data()

    pd.testing.assert_frame_equal(loaded_features, features)
    pd.testing.assert_frame_equal(loaded_target, target)