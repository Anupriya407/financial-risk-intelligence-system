import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import optuna

from app.ml.evaluation import evaluate_classifier
from app.ml.models.factory import create_model
from app.ml.preprocessing.preprocessor import MLPreprocessor
from app.ml.splitting.holdout import create_holdout_split


FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)

TARGET_PATH = Path(
    "data/processed/features/fris_training_target.parquet"
)

LIGHTGBM_RESULT_PATH = Path(
    "data/processed/ml/optimization_result.json"
)

XGBOOST_RESULT_PATH = Path(
    "data/processed/ml/xgboost_optimization_result.json"
)

RESULT_PATH = Path(
    "data/processed/ml/final_model_comparison.json"
)


class FrozenTrialAdapter:
    """Minimal Optuna trial adapter for saved parameters."""

    def __init__(
        self,
        params: dict[str, Any],
    ) -> None:
        self.params = params

    def suggest_categorical(
        self,
        name: str,
        choices: list[Any],
    ) -> Any:
        return self.params[name]

    def suggest_float(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        return self.params[name]

    def suggest_int(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        return self.params[name]


def build_pipeline(
    params: dict[str, Any],
) -> Pipeline:
    """Build a model pipeline from saved Optuna parameters."""

    trial = FrozenTrialAdapter(params)

    model = create_model(trial)

    return Pipeline(
        steps=[
            (
                "preprocessor",
                clone(
                    MLPreprocessor().pipeline
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def load_best_params(
    path: Path,
) -> dict[str, Any]:
    """Load saved optimization parameters."""

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return data["best_params"]


def main() -> None:
    """Compare LightGBM and XGBoost on the untouched holdout."""

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
            "Features and target must contain "
            "the same number of rows."
        )

    holdout = create_holdout_split(
        features=features,
        target=target,
        test_size=0.15,
        random_state=42,
    )

    print("=" * 60)
    print("FINAL HOLDOUT MODEL COMPARISON")
    print("=" * 60)

    print(
        f"Development rows: "
        f"{len(holdout.X_development)}"
    )

    print(
        f"Final test rows: "
        f"{len(holdout.X_test)}"
    )

    print()
    print(
        "The final test set will NOT be used "
        "during training."
    )
    print()

    lightgbm_params = load_best_params(
        LIGHTGBM_RESULT_PATH
    )

    xgboost_params = load_best_params(
        XGBOOST_RESULT_PATH
    )

    models = {
        "lightgbm": build_pipeline(
            lightgbm_params
        ),
        "xgboost": build_pipeline(
            xgboost_params
        ),
    }

    results: dict[str, dict[str, float]] = {}

    for model_name, pipeline in models.items():

        print(
            f"Training {model_name.upper()} "
            f"on development data..."
        )

        pipeline.fit(
            holdout.X_development,
            holdout.y_development,
        )

        metrics = evaluate_classifier(
            model=pipeline,
            X_test=holdout.X_test,
            y_test=holdout.y_test,
        )

        results[model_name] = metrics

        print()
        print(
            f"{model_name.upper()} RESULTS"
        )
        print("-" * 40)

        for metric, value in metrics.items():
            print(
                f"{metric:10s}: {value:.6f}"
            )

        print()

    lightgbm_auc = results[
        "lightgbm"
    ]["roc_auc"]

    xgboost_auc = results[
        "xgboost"
    ]["roc_auc"]

    if xgboost_auc > lightgbm_auc:
        winner = "xgboost"
    else:
        winner = "lightgbm"

    results["winner"] = winner

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("FINAL MODEL COMPARISON COMPLETE")
    print("=" * 60)

    print(
        f"LightGBM holdout ROC-AUC: "
        f"{lightgbm_auc:.6f}"
    )

    print(
        f"XGBoost holdout ROC-AUC:  "
        f"{xgboost_auc:.6f}"
    )

    print(
        f"FINAL WINNER: {winner.upper()}"
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()