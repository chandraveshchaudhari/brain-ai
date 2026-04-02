"""Registry-driven time-series executor."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from brain_automl.config import get_default_config
from brain_automl.core.protocols import BaseModalityExecutor
from brain_automl.core.registry import BACKEND_REGISTRY
from brain_automl.core.result import ModalityResult
from brain_automl.model_zoo.time_series_ai import backends  # noqa: F401
from brain_automl.model_zoo.time_series_ai.data_preparation import (
    DEFAULT_BUSINESS_SEASONALITY,
    compute_forecast_metrics,
    infer_frequency,
    normalize_prediction_frame,
    regularize_series_frequency,
    select_item_series,
    split_last_horizon,
    to_standard_timeseries_format,
)


class TimeSeriesAutoML(BaseModalityExecutor):
    """Runs configured time-series backends and normalizes outputs."""

    modality = "time_series"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or get_default_config()

    def _selected_backends(self) -> List[str]:
        backend_cfg = self.config["backends"]["by_modality"]["time_series"]
        return list(backend_cfg.get("default", []))

    def available_backends(self, backends: Optional[Iterable[str]] = None) -> List[str]:
        """Return registered, import-available backends for time-series execution."""
        selected = list(backends) if backends is not None else self._selected_backends()
        available: List[str] = []
        for backend_name in selected:
            if not BACKEND_REGISTRY.has(backend_name):
                continue
            backend_cls = BACKEND_REGISTRY.get(backend_name)
            if hasattr(backend_cls, "modality") and backend_cls.modality != self.modality:
                continue
            if backend_cls.is_available():
                available.append(backend_name)
        return available

    def run(
        self,
        data: pd.DataFrame,
        task: str = "forecasting",
        backends: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[ModalityResult]:
        selected = list(backends) if backends is not None else self._selected_backends()
        outputs: List[ModalityResult] = []

        for backend_name in selected:
            if not BACKEND_REGISTRY.has(backend_name):
                if self.config["backends"]["skip_unavailable_backends"]:
                    continue
                raise KeyError(f"Time-series backend '{backend_name}' is not registered")

            backend_cls = BACKEND_REGISTRY.get(backend_name)
            if hasattr(backend_cls, "modality") and backend_cls.modality != self.modality:
                continue

            if not backend_cls.is_available():
                if self.config["backends"]["skip_unavailable_backends"]:
                    continue
                raise RuntimeError(f"Backend '{backend_name}' is not available in this environment")

            backend = backend_cls()
            model = backend.fit(data, None, task=task, **kwargs)
            prediction = backend.predict(model, data, task=task, **kwargs)
            outputs.append(
                ModalityResult(
                    modality=self.modality,
                    backend=backend_name,
                    task=task,
                    predictions=prediction,
                    metrics={},
                    metadata={"status": "ok"},
                )
            )

        if not outputs and self.config["backends"]["hard_fail_if_no_backend_available_for_modality"]:
            raise RuntimeError("No available time-series backend produced output")

        return outputs

    def forecast_last_horizon(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        target_column: str,
        item_id_column: Optional[str] = None,
        item_id: Optional[str] = None,
        horizon: int = 252,
        backends: Optional[Iterable[str]] = None,
        frequency: Optional[str] = None,
        seasonality: int = DEFAULT_BUSINESS_SEASONALITY,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run all selected time-series backends on a last-horizon holdout split.

        Returns a dictionary with normalized predictions, metrics, raw modality
        results, and train/test splits so notebooks can consume the source API
        directly without reimplementing benchmarking logic.
        """
        standard_df = to_standard_timeseries_format(
            data,
            target_column=target_column,
            timestamp_column=timestamp_column,
            item_id_column=item_id_column,
        )
        series_df, selected_item = select_item_series(standard_df, item_id=item_id)
        inferred_frequency = frequency or infer_frequency(series_df["ds"])
        series_df = regularize_series_frequency(series_df, inferred_frequency)
        train_df, test_df = split_last_horizon(series_df, horizon=horizon)
        selected_backends = self.available_backends(backends)

        prediction_frame = test_df[["ds", "y"]].rename(columns={"y": "actual"}).copy()
        results: List[ModalityResult] = []
        metric_rows: List[Dict[str, Any]] = []

        for backend_name in selected_backends:
            backend_cls = BACKEND_REGISTRY.get(backend_name)
            backend = backend_cls()
            try:
                model = backend.fit(
                    train_df,
                    None,
                    task="forecasting",
                    prediction_length=len(test_df),
                    horizon=len(test_df),
                    frequency=inferred_frequency,
                    seasonality=seasonality,
                    **kwargs,
                )
                raw_prediction = backend.predict(
                    model,
                    test_df,
                    task="forecasting",
                    prediction_length=len(test_df),
                    horizon=len(test_df),
                    frequency=inferred_frequency,
                    seasonality=seasonality,
                    **kwargs,
                )
                normalized_prediction = normalize_prediction_frame(
                    raw_prediction,
                    backend_name=backend_name,
                    expected_dates=test_df["ds"],
                )
                prediction_frame[backend_name] = normalized_prediction["prediction"].to_numpy()
                metrics = compute_forecast_metrics(
                    y_train=train_df["y"],
                    y_true=test_df["y"],
                    y_pred=normalized_prediction["prediction"],
                    seasonality=seasonality,
                )
                results.append(
                    ModalityResult(
                        modality=self.modality,
                        backend=backend_name,
                        task="forecasting",
                        predictions=normalized_prediction,
                        metrics=metrics,
                        metadata={
                            "status": "ok",
                            "item_id": selected_item,
                            "frequency": inferred_frequency,
                            "horizon": len(test_df),
                        },
                    )
                )
                metric_rows.append({"backend": backend_name, **metrics})
            except Exception as exc:
                results.append(
                    ModalityResult(
                        modality=self.modality,
                        backend=backend_name,
                        task="forecasting",
                        predictions=None,
                        metrics={},
                        metadata={
                            "status": "error",
                            "item_id": selected_item,
                            "frequency": inferred_frequency,
                            "horizon": len(test_df),
                            "error": str(exc),
                        },
                    )
                )
                if not self.config["modalities"]["allow_partial_success"]:
                    raise

        if not metric_rows and self.config["backends"]["hard_fail_if_no_backend_available_for_modality"]:
            raise RuntimeError("No available time-series backend produced a forecast")

        metrics_df = pd.DataFrame(metric_rows)
        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values("rmse").reset_index(drop=True)

        return {
            "item_id": selected_item,
            "frequency": inferred_frequency,
            "horizon": len(test_df),
            "train_data": train_df,
            "test_data": test_df,
            "predictions": prediction_frame,
            "metrics": metrics_df,
            "results": results,
        }
