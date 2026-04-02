"""FLAML time-series backend adapter."""

from __future__ import annotations

from typing import Any

from brain_automl.core.protocols import BaseLibraryBackend
from brain_automl.core.registry import BACKEND_REGISTRY


@BACKEND_REGISTRY.register("flaml")
class FLAMLTimeSeriesBackend(BaseLibraryBackend):
    name = "flaml"
    modality = "time_series"
    task_types = ("forecasting",)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import flaml  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, x_train: Any, y_train: Any = None, **kwargs: Any) -> Any:
        return {"backend": self.name, "kwargs": kwargs}

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        return {"message": "flaml adapter scaffolded", "model": model}
