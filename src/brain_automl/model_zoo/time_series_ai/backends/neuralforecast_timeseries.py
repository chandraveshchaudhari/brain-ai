"""NeuralForecast backend adapter."""

from __future__ import annotations

from typing import Any

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from brain_automl.core.protocols import BaseLibraryBackend
from brain_automl.core.registry import BACKEND_REGISTRY
from brain_automl.model_zoo.time_series_ai.data_preparation import to_neuralforecast_format


@BACKEND_REGISTRY.register("neuralforecast")
class NeuralForecastBackend(BaseLibraryBackend):
    name = "neuralforecast"
    modality = "time_series"
    task_types = ("forecasting",)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import neuralforecast  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, x_train: Any, y_train: Any = None, **kwargs: Any) -> Any:
        prediction_length = int(kwargs.get("prediction_length") or kwargs.get("horizon") or 14)
        frequency = kwargs.get("frequency", "B")
        max_steps = int(kwargs.get("max_steps", 5))
        input_size = int(kwargs.get("input_size", min(max(24, prediction_length // 2), 64)))
        max_allowed_val_size = max(1, len(x_train) - 1)
        val_size = int(kwargs.get("val_size", min(max(8, prediction_length // 6), 32, max_allowed_val_size)))

        train_data = to_neuralforecast_format(x_train)
        model = NHITS(
            h=prediction_length,
            input_size=input_size,
            max_steps=max_steps,
            val_check_steps=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        forecaster = NeuralForecast(models=[model], freq=frequency)
        forecaster.fit(df=train_data, val_size=val_size)
        return {
            "backend": self.name,
            "forecaster": forecaster,
            "prediction_length": prediction_length,
            "model_name": model.__class__.__name__,
        }

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        forecast_df = model["forecaster"].predict()
        model_name = model["model_name"]
        return forecast_df[["unique_id", "ds", model_name]].rename(columns={model_name: "prediction"})