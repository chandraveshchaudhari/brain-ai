from .base import BaseGranularity
from typing import Any
import numpy as np


class PoolingGranularity(BaseGranularity):
    def __init__(self, window: int = 5, method: str = "mean"):
        self.window = max(1, int(window))
        self.method = method

    def align(self, data: Any) -> Any:
        modalities = data["modalities"]
        aligned = {}
        for name, values in modalities.items():
            arr = np.asarray(values)
            pooled_rows = []
            for start in range(0, arr.shape[0], self.window):
                chunk = arr[start : start + self.window]
                if chunk.size == 0:
                    continue
                if self.method == "max":
                    pooled_rows.append(chunk.max(axis=0))
                else:
                    pooled_rows.append(chunk.mean(axis=0))
            aligned[name] = np.asarray(pooled_rows)

        target_len = min(v.shape[0] for v in aligned.values())
        return {
            "modalities": {name: values[:target_len] for name, values in aligned.items()},
            "y": np.asarray(data["y"])[:target_len],
        }
