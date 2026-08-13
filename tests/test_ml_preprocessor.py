import numpy as np
import pandas as pd

from app.ml.preprocessing.preprocessor import MLPreprocessor


def test_preprocessor_imputes_missing_values() -> None:
    features = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.nan, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        }
    )

    preprocessor = MLPreprocessor()

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape == (4, 2)
    assert not np.isnan(transformed).any()


def test_preprocessor_standardizes_features() -> None:
    features = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        }
    )

    preprocessor = MLPreprocessor()

    transformed = preprocessor.fit_transform(features)

    assert np.allclose(transformed.mean(axis=0), 0.0)
    assert np.allclose(transformed.std(axis=0), 1.0)


def test_preprocessor_transform_uses_fitted_pipeline() -> None:
    train = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, 20.0, 30.0],
        }
    )

    validation = pd.DataFrame(
        {
            "feature_a": [4.0, 5.0],
            "feature_b": [40.0, 50.0],
        }
    )

    preprocessor = MLPreprocessor()
    preprocessor.fit(train)

    transformed = preprocessor.transform(validation)

    assert transformed.shape == (2, 2)
    assert np.isfinite(transformed).all()

def test_preprocessor_handles_categorical_features() -> None:
    features = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0, None, 4.0],
            "category_feature": [
                "Cash loans",
                "Revolving loans",
                "Cash loans",
                None,
            ],
        }
    )

    preprocessor = MLPreprocessor()

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] > 2

    if hasattr(transformed, "toarray"):
        transformed_array = transformed.toarray()
    else:
        transformed_array = transformed

    assert np.isfinite(transformed_array).all()