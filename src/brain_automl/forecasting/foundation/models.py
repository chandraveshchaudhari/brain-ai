"""Foundation forecasting model registry with optional dependency checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FoundationModelSpec:
    """Definition of one foundation forecasting model."""

    name: str
    import_path: str
    supports_multivariate: bool
    provider: str

    def is_available(self) -> bool:
        try:
            __import__(self.import_path)
            return True
        except Exception:
            return False


FOUNDATION_MODELS: Dict[str, FoundationModelSpec] = {
    "chronos": FoundationModelSpec(
        name="chronos",
        import_path="chronos",
        supports_multivariate=True,
        provider="amazon",
    ),
    "timesfm": FoundationModelSpec(
        name="timesfm",
        import_path="timesfm",
        supports_multivariate=True,
        provider="google",
    ),
    "lag_llama": FoundationModelSpec(
        name="lag_llama",
        import_path="lag_llama",
        supports_multivariate=True,
        provider="servicenow",
    ),
    "moirai": FoundationModelSpec(
        name="moirai",
        import_path="uni2ts",
        supports_multivariate=True,
        provider="salesforce",
    ),
}


def list_foundation_models(include_unavailable: bool = False) -> List[Dict[str, object]]:
    """Return model catalog rows in a unified format."""
    rows: List[Dict[str, object]] = []
    for model_name, spec in FOUNDATION_MODELS.items():
        available = spec.is_available()
        if not include_unavailable and not available:
            continue
        rows.append(
            {
                "model": model_name,
                "provider": spec.provider,
                "supports_multivariate": spec.supports_multivariate,
                "available": available,
            }
        )
    return rows


def get_foundation_model_spec(model_name: str) -> Optional[FoundationModelSpec]:
    """Return model definition for a foundation model key."""
    return FOUNDATION_MODELS.get(model_name)
