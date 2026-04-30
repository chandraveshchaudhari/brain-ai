from .base import BaseFusion
from typing import Any
import numpy as np


class EarlyFusion(BaseFusion):
    def fuse(self, features: Any) -> Any:
        modalities = features["modalities"]
        names = sorted(modalities.keys())
        X = np.concatenate([np.asarray(modalities[name]) for name in names], axis=1)
        return {"X": X, "y": np.asarray(features["y"])}
