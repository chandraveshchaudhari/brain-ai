"""AutoGluon TimeSeries backend adapter."""

from __future__ import annotations

from typing import Any

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from brain_automl.core.protocols import BaseLibraryBackend
from brain_automl.core.registry import BACKEND_REGISTRY
from brain_automl.model_zoo.time_series_ai.data_preparation import to_autogluon_timeseries_format


@BACKEND_REGISTRY.register("autogluon_timeseries")
class AutoGluonTimeSeriesBackend(BaseLibraryBackend):
    name = "autogluon_timeseries"
    modality = "time_series"
    task_types = ("forecasting",)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import autogluon.timeseries  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, x_train: Any, y_train: Any = None, **kwargs: Any) -> Any:
        prediction_length = int(kwargs.get("prediction_length") or kwargs.get("horizon") or 14)
        eval_metric = kwargs.get("eval_metric", "MASE")
        presets = kwargs.get("presets", "fast_training")
        time_limit = kwargs.get("time_limit", 60)
        verbosity = kwargs.get("verbosity", 0)
        output_dir = kwargs.get("output_dir")

        ag_train = to_autogluon_timeseries_format(
            x_train,
            target_column="y",
            timestamp_column="ds",
            item_id_column="unique_id",
        )
        train_data = TimeSeriesDataFrame.from_data_frame(ag_train)
        predictor_kwargs: dict = dict(
            prediction_length=prediction_length,
            target="target",
            eval_metric=eval_metric,
            verbosity=verbosity,
        )
        if output_dir:
            predictor_kwargs["path"] = output_dir
        predictor = TimeSeriesPredictor(**predictor_kwargs)
        predictor.fit(train_data, presets=presets, time_limit=time_limit)
        return {
            "backend": self.name,
            "predictor": predictor,
            "train_data": train_data,
            "prediction_length": prediction_length,
        }

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        forecast = model["predictor"].predict(model["train_data"])
        forecast_df = forecast.reset_index()
        return forecast_df[["item_id", "timestamp", "mean"]].rename(
            columns={"item_id": "unique_id", "timestamp": "ds", "mean": "prediction"}
        )