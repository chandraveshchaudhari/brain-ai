"""StatsForecast backend adapter."""

from __future__ import annotations

from typing import Any

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive, SeasonalNaive

from brain_automl.core.protocols import BaseLibraryBackend
from brain_automl.core.registry import BACKEND_REGISTRY


@BACKEND_REGISTRY.register("statsforecast")
class StatsForecastBackend(BaseLibraryBackend):
    name = "statsforecast"
    modality = "time_series"
    task_types = ("forecasting",)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import statsforecast  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, x_train: Any, y_train: Any = None, **kwargs: Any) -> Any:
        frequency = kwargs.get("frequency", "B")
        seasonality = int(kwargs.get("seasonality", 5))
        prediction_length = int(kwargs.get("prediction_length") or kwargs.get("horizon") or 14)
        model_names = ["AutoARIMA", "SeasonalNaive", "Naive"]
        forecaster = StatsForecast(
            models=[AutoARIMA(season_length=seasonality), SeasonalNaive(season_length=seasonality), Naive()],
            freq=frequency,
            n_jobs=1,
            fallback_model=Naive(),
        )
        return {
            "backend": self.name,
            "forecaster": forecaster,
            "train_data": x_train[["unique_id", "ds", "y"]].copy(),
            "prediction_length": prediction_length,
            "primary_model_name": model_names[0],
        }

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        forecast_df = model["forecaster"].forecast(
            df=model["train_data"],
            h=model["prediction_length"],
        )
        primary_model_name = model["primary_model_name"]
        return forecast_df[["unique_id", "ds", primary_model_name]].rename(columns={primary_model_name: "prediction"})