from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


@dataclass(frozen=True)
class PipelineSpec:
    """Serializable pipeline definition shared across all layers."""

    modalities: List[str]
    granularity_strategy: str
    fusion_strategy: str
    model_backend: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineSpec":
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "PipelineSpec":
        return cls.from_dict(json.loads(payload))

    def log_mlflow(self) -> None:
        if mlflow is None:
            return
        mlflow.log_params(
            {
                "modalities": ",".join(self.modalities),
                "granularity_strategy": self.granularity_strategy,
                "fusion_strategy": self.fusion_strategy,
                "model_backend": self.model_backend,
                "pipeline_spec_json": self.to_json(),
            }
        )
        if self.hyperparameters:
            mlflow.log_param("hyperparameters", json.dumps(self.hyperparameters, sort_keys=True))
