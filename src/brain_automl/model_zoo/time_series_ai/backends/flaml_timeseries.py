"""FLAML time-series backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from flaml import AutoML

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
        seasonality = int(kwargs.get("seasonality", 5))
        time_budget = int(kwargs.get("time_budget", 30))
        metric = kwargs.get("metric", "mape")
        output_dir = kwargs.get("output_dir")

        log_file_name = str(Path(output_dir) / "flaml.log") if output_dir else None

        automl = AutoML()
        train_x = x_train[["ds"]].copy()
        train_y = x_train["y"].copy()
        automl.fit(
            X_train=train_x,
            y_train=train_y,
            task="ts_forecast_regression",
            period=seasonality,
            time_budget=time_budget,
            metric=metric,
            verbose=0,
            log_file_name=log_file_name,
        )
        return {
            "backend": self.name,
            "automl": automl,
        }

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        prediction_values = model["automl"].predict(x_test[["ds"]].copy())
        return pd.DataFrame({
            "unique_id": x_test["unique_id"].values,
            "ds": pd.to_datetime(x_test["ds"].values),
            "prediction": prediction_values,
        })