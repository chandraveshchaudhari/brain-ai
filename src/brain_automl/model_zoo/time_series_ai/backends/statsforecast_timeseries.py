"""StatsForecast backend adapter."""

from __future__ import annotations

from typing import Any, List, Tuple

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

    @staticmethod
    def _resolve_models(model_names: List[str], seasonality: int) -> Tuple[List[Any], str]:
        """Build StatsForecast model objects from model names.

        Returns model list and selected primary model name.
        """
        from statsforecast import models as sf_models

        built_models: List[Any] = []
        selected_name = ""
        for model_name in model_names:
            key = str(model_name).strip()
            cls = getattr(sf_models, key, None)
            if cls is None:
                continue
            if key in {"AutoARIMA", "SeasonalNaive", "Naive"}:
                if key == "AutoARIMA":
                    built_models.append(cls(season_length=seasonality))
                elif key == "SeasonalNaive":
                    built_models.append(cls(season_length=seasonality))
                else:
                    built_models.append(cls())
            elif key in {"AutoETS", "ETS"}:
                # AutoETS is generally available; ETS may exist in some versions.
                built_models.append(cls(season_length=seasonality))
            else:
                try:
                    built_models.append(cls())
                except TypeError:
                    built_models.append(cls(season_length=seasonality))

            if not selected_name:
                selected_name = key

        if not built_models:
            if model_names:
                raise ValueError(
                    f"No requested StatsForecast models are available in installed version: {model_names}"
                )
            built_models = [AutoARIMA(season_length=seasonality), SeasonalNaive(season_length=seasonality), Naive()]
            selected_name = "AutoARIMA"
        return built_models, selected_name

    def fit(self, x_train: Any, y_train: Any = None, **kwargs: Any) -> Any:
        frequency = kwargs.get("frequency", "B")
        seasonality = int(kwargs.get("seasonality", 5))
        prediction_length = int(kwargs.get("prediction_length") or kwargs.get("horizon") or 14)
        single_model = kwargs.get("statsforecast_model")
        model_names = kwargs.get("statsforecast_models")
        if model_names is None:
            model_names = [single_model] if single_model else ["AutoARIMA", "SeasonalNaive", "Naive"]

        resolved_models, primary_model_name = self._resolve_models(list(model_names), seasonality)
        forecaster = StatsForecast(
            models=resolved_models,
            freq=frequency,
            n_jobs=1,
            fallback_model=Naive(),
        )

        # Ensure x_train is a proper DataFrame with required columns.
        import pandas as _pd
        train_df = _pd.DataFrame(x_train) if not isinstance(x_train, _pd.DataFrame) else x_train.copy()

        # StatsForecast requires unique_id, ds, y as plain columns.
        if "unique_id" not in train_df.columns:
            if train_df.index.name == "unique_id":
                train_df = train_df.reset_index()
            else:
                train_df["unique_id"] = "series_0"

        if "ds" not in train_df.columns:
            # Try common timestamp column names
            for candidate in ("timestamp", "date", "Date"):
                if candidate in train_df.columns:
                    train_df = train_df.rename(columns={candidate: "ds"})
                    break

        if "y" not in train_df.columns:
            # Try common target column names
            for candidate in ("target", "value", "close", "Close"):
                if candidate in train_df.columns:
                    train_df = train_df.rename(columns={candidate: "y"})
                    break

        return {
            "backend": self.name,
            "forecaster": forecaster,
            "train_data": train_df[["unique_id", "ds", "y"]].copy(),
            "prediction_length": prediction_length,
            "primary_model_name": primary_model_name,
            "configured_models": list(model_names),
        }

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        import pandas as _pd

        forecast_df = model["forecaster"].forecast(
            df=model["train_data"],
            h=model["prediction_length"],
        )

        # statsforecast ≥1.7 returns unique_id+ds as a MultiIndex or unique_id as index.
        # Normalise to plain columns so downstream code is uniform.
        if forecast_df.index.name == "unique_id" or (
            hasattr(forecast_df.index, "names") and "unique_id" in forecast_df.index.names
        ):
            forecast_df = forecast_df.reset_index()

        # Ensure ds is a column (may be index on some versions).
        if "ds" not in forecast_df.columns:
            forecast_df = forecast_df.reset_index()

        primary_model_name = model["primary_model_name"]
        if primary_model_name in forecast_df.columns:
            selected_col = primary_model_name
        else:
            candidate_cols = [c for c in forecast_df.columns if c not in {"unique_id", "ds"}]
            if not candidate_cols:
                raise ValueError("StatsForecast returned no prediction columns")
            selected_col = candidate_cols[0]

        result = forecast_df[["ds", selected_col]].rename(columns={selected_col: "prediction"}).copy()
        if "unique_id" in forecast_df.columns:
            result["unique_id"] = forecast_df["unique_id"].values
        else:
            result["unique_id"] = model["train_data"]["unique_id"].iloc[0]

        return result[["unique_id", "ds", "prediction"]]