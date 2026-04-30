from .base import BaseFusion
from typing import Any
import numpy as np


class IntermediateFusion(BaseFusion):
    def fuse(self, features: Any) -> Any:
        modalities = features["modalities"]
        names = sorted(modalities.keys())
        projected = []
        for name in names:
            arr = np.asarray(modalities[name])
            projected.append(np.tanh(arr))

        base = np.concatenate(projected, axis=1)
        interaction = None
        if len(projected) > 1:
            interaction = projected[0][:, :1] * projected[1][:, :1]
        if interaction is None:
            interaction = np.zeros((base.shape[0], 1), dtype=base.dtype)

        X = np.concatenate([base, interaction], axis=1)
        return {"X": X, "y": np.asarray(features["y"])}
