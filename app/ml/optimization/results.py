from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import optuna


@dataclass(frozen=True)
class OptimizationResult:
    """Serializable summary of an Optuna optimization."""

    model_type: str
    best_score: float
    best_params: dict[str, object]
    n_trials: int
    cv_scores: list[float]
    completed_at: str

    @classmethod
    def from_study(cls, study: optuna.Study) -> "OptimizationResult":
        """Create an optimization result from a completed study."""

        if not study.trials:
            raise ValueError("Study contains no trials.")

        best_trial = study.best_trial

        cv_scores = best_trial.user_attrs.get("cv_scores", [])

        return cls(
            model_type=str(best_trial.params["model_type"]),
            best_score=float(study.best_value),
            best_params=dict(best_trial.params),
            n_trials=len(study.trials),
            cv_scores=[float(score) for score in cv_scores],
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def save(self, path: str | Path) -> None:
        """Save optimization results as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(asdict(self), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "OptimizationResult":
        """Load optimization results from JSON."""

        input_path = Path(path)

        data = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        return cls(**data)