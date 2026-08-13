from pathlib import Path

from app.ml.optimization.evaluation_results import EvaluationResult


def test_evaluation_result_save_and_load(tmp_path: Path) -> None:
    result = EvaluationResult(
        accuracy=0.85,
        precision=0.82,
        recall=0.78,
        f1=0.80,
        roc_auc=0.91,
    )

    path = tmp_path / "evaluation.json"

    result.save(path)

    assert path.exists()

    loaded = EvaluationResult.load(path)

    assert loaded == result