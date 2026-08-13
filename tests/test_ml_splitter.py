import pandas as pd
import pytest

from app.ml.splitting.splitter import DatasetSplitter


def create_dataset() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "feature_a": range(1000),
            "feature_b": range(1000, 2000),
        }
    )

    target = pd.Series(
        [0] * 800 + [1] * 200,
        name="TARGET",
    )

    return features, target


def test_split_creates_three_sets() -> None:
    features, target = create_dataset()

    splitter = DatasetSplitter(
        test_size=0.15,
        validation_size=0.15,
        random_state=42,
    )

    result = splitter.split(features, target)

    assert len(result.X_train) == 700
    assert len(result.X_validation) == 150
    assert len(result.X_test) == 150

    assert len(result.y_train) == 700
    assert len(result.y_validation) == 150
    assert len(result.y_test) == 150


def test_split_is_reproducible() -> None:
    features, target = create_dataset()

    splitter = DatasetSplitter(random_state=42)

    first = splitter.split(features, target)
    second = splitter.split(features, target)

    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_frame_equal(first.X_validation, second.X_validation)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)


def test_split_preserves_target_distribution() -> None:
    features, target = create_dataset()

    splitter = DatasetSplitter(random_state=42)

    result = splitter.split(features, target)

    assert result.y_train.mean() == pytest.approx(0.20, abs=0.01)
    assert result.y_validation.mean() == pytest.approx(0.20, abs=0.02)
    assert result.y_test.mean() == pytest.approx(0.20, abs=0.02)


def test_split_rejects_mismatched_lengths() -> None:
    features, target = create_dataset()

    splitter = DatasetSplitter()

    with pytest.raises(ValueError, match="same number of rows"):
        splitter.split(features, target.iloc[:-1])


def test_split_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        DatasetSplitter(test_size=0)

    with pytest.raises(ValueError):
        DatasetSplitter(validation_size=0)

    with pytest.raises(ValueError):
        DatasetSplitter(test_size=0.6, validation_size=0.5)