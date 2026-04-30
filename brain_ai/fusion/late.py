from .base import BaseFusion
from typing import Any
import numpy as np


class LateFusion(BaseFusion):
    def fuse(self, features: Any) -> Any:
        modalities = features["modalities"]
        names = sorted(modalities.keys())
        blocks = []
        for name in names:
            arr = np.asarray(modalities[name])
            mean_feature = arr.mean(axis=1, keepdims=True)
            std_feature = arr.std(axis=1, keepdims=True)
            blocks.append(np.concatenate([mean_feature, std_feature], axis=1))
        X = np.concatenate(blocks, axis=1)
        return {"X": X, "y": np.asarray(features["y"])}
