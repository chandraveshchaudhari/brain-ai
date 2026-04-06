"""Deep learning forecasting model catalog used for experiment planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DeepLearningModelSpec:
    """Spec for one deep-learning forecasting architecture."""

    name: str
    preferred_library: str
    multivariate_capable: bool


DEEP_LEARNING_MODELS: Dict[str, DeepLearningModelSpec] = {
    "tcn": DeepLearningModelSpec("tcn", "darts", True),
    "patchtst": DeepLearningModelSpec("patchtst", "neuralforecast", True),
    "nhits": DeepLearningModelSpec("nhits", "neuralforecast", True),
    "nbeats": DeepLearningModelSpec("nbeats", "neuralforecast", True),
    "tide": DeepLearningModelSpec("tide", "neuralforecast", True),
    "itransformer": DeepLearningModelSpec("itransformer", "neuralforecast", True),
}


def list_deep_learning_models() -> List[Dict[str, object]]:
    """Return unified deep-learning model metadata."""
    return [
        {
            "model": spec.name,
            "preferred_library": spec.preferred_library,
            "multivariate_capable": spec.multivariate_capable,
        }
        for spec in DEEP_LEARNING_MODELS.values()
    ]
