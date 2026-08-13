import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from app.ml.preprocessing.preprocessor import MLPreprocessor


FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)

TARGET_PATH = Path(
    "data/processed/features/fris_training_target.parquet"
)

OPTIMIZATION_RESULT_PATH = Path(
    "data/processed/ml/xgboost_optimization_result.json"
)

ARTIFACT_PATH = Path(
    "data/processed/ml/best_model.joblib"
)


def load_xgboost_parameters() -> dict[str, Any]:
    """Load the winning XGBoost parameters."""

    data = json.loads(
        OPTIMIZATION_RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )

    params = data["best_params"]

    if params.get("model_type") != "xgboost":
        raise ValueError(
            "Saved optimization result is not for XGBoost."
        )

    return params


def build_xgboost_model(
    params: dict[str, Any],
) -> XGBClassifier:
    """Build XGBoost from the saved Optuna parameters."""

    return XGBClassifier(
        n_estimators=params["xgb_n_estimators"],
        max_depth=params["xgb_max_depth"],
        learning_rate=params["xgb_learning_rate"],
        subsample=params["xgb_subsample"],
        colsample_bytree=params["xgb_colsample_bytree"],
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )


def main() -> None:
    """Train and persist the final FRIS V1 XGBoost model."""

    features = pd.read_parquet(
        FEATURES_PATH
    )

    target_df = pd.read_parquet(
        TARGET_PATH
    )

    if "TARGET" not in target_df.columns:
        raise ValueError(
            "TARGET column not found."
        )

    target = target_df["TARGET"]

    if len(features) != len(target):
        raise ValueError(
            "Features and target must have "
            "the same number of rows."
        )

    params = load_xgboost_parameters()

    model = build_xgboost_model(
        params
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                MLPreprocessor().pipeline,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print("=" * 60)
    print("FINAL FRIS V1 MODEL TRAINING")
    print("=" * 60)

    print(
        f"Training rows: {len(features)}"
    )

    print(
        f"Feature count: {features.shape[1]}"
    )

    print("Model: XGBoost")
    print()

    print("Winning parameters:")

    for key, value in params.items():
        print(
            f"  {key}: {value}"
        )

    print()
    print("Training final model...")

    pipeline.fit(
        features,
        target,
    )

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        ARTIFACT_PATH,
    )

    print()
    print("=" * 60)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Model: XGBoost"
    )

    print(
        f"Artifact: {ARTIFACT_PATH}"
    )

    print(
        f"File exists: {ARTIFACT_PATH.exists()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()