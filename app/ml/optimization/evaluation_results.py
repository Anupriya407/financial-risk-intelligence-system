from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class EvaluationResult:
    """Final holdout evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def save(self, path: str | Path) -> None:
        """Save evaluation metrics as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(asdict(self), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationResult":
        """Load evaluation metrics from JSON."""

        input_path = Path(path)

        data = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        return cls(**data)