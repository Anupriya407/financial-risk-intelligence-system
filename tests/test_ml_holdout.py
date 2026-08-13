import pandas as pd
import pytest

from app.ml.splitting.holdout import create_holdout_split


def test_create_holdout_split() -> None:
    features = pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": range(100, 200),
        }
    )

    target = pd.Series([0] * 80 + [1] * 20)

    result = create_holdout_split(
        features,
        target,
        test_size=0.20,
    )

    assert len(result.X_development) == 80
    assert len(result.X_test) == 20
    assert len(result.y_development) == 80
    assert len(result.y_test) == 20


def test_holdout_preserves_target_distribution() -> None:
    features = pd.DataFrame({"feature": range(100)})
    target = pd.Series([0] * 80 + [1] * 20)

    result = create_holdout_split(
        features,
        target,
        test_size=0.20,
    )

    assert result.y_test.mean() == pytest.approx(0.20, abs=0.01)
    assert result.y_development.mean() == pytest.approx(0.20, abs=0.01)