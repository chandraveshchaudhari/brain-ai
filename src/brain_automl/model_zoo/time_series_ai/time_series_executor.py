"""Registry-driven time-series executor."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from brain_automl.config import get_default_config
from brain_automl.core.protocols import BaseModalityExecutor
from brain_automl.core.registry import BACKEND_REGISTRY
from brain_automl.core.result import ModalityResult
from brain_automl.model_zoo.time_series_ai import backends  # noqa: F401
from brain_automl.model_zoo.time_series_ai.backtesting import expanding_window_backtest
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
from brain_automl.data_processing import split_data_by_stock
from brain_automl.utilities.run_logging import setup_run_logger


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

    def _extract_backend_model_info(self, backend_name: str, fitted_model: Any) -> Dict[str, Any]:
        """Best-effort extraction of selected and trained model names per backend."""
        selected_model: Optional[str] = None
        trained_models: List[str] = []

        if backend_name == "autogluon_timeseries" and isinstance(fitted_model, dict):
            predictor = fitted_model.get("predictor")
            if predictor is not None:
                try:
                    names = predictor.model_names()
                    trained_models = [str(n) for n in names]
                except Exception:
                    trained_models = []
                try:
                    leaderboard = predictor.leaderboard(silent=True)
                    if leaderboard is not None and not leaderboard.empty and "model" in leaderboard.columns:
                        selected_model = str(leaderboard.iloc[0]["model"])
                except Exception:
                    selected_model = trained_models[0] if trained_models else None

        elif backend_name == "statsforecast" and isinstance(fitted_model, dict):
            selected_model = str(fitted_model.get("primary_model_name") or "") or None
            configured = fitted_model.get("configured_models") or []
            trained_models = [str(m) for m in configured if str(m).strip()]

        elif backend_name == "flaml" and isinstance(fitted_model, dict):
            automl = fitted_model.get("automl")
            if automl is not None:
                best_estimator = getattr(automl, "best_estimator", None)
                if best_estimator is not None:
                    selected_model = str(best_estimator)
                    trained_models = [selected_model]

        elif backend_name == "chronos" and isinstance(fitted_model, dict):
            selected_model = str(fitted_model.get("model_name") or "amazon/chronos-t5-small")
            trained_models = [selected_model]

        if selected_model is None:
            selected_model = backend_name
        if not trained_models:
            trained_models = [selected_model]

        return {
            "selected_model": selected_model,
            "trained_models": trained_models,
        }

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
        include_comprehensive_experiments: bool = False,
        comprehensive_forecast_columns: Optional[Iterable[str]] = None,
        comprehensive_decomposition_types: Optional[Iterable[str]] = None,
        comprehensive_model_types: Optional[Iterable[str]] = None,
        comprehensive_train_ratio: float = 0.8,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run all selected time-series backends on a last-horizon holdout split.

        Returns a dictionary with normalized predictions, metrics, raw modality
        results, and train/test splits so notebooks can consume the source API
        directly without reimplementing benchmarking logic.
        """
        kwargs = dict(kwargs)
        # --- Output directory & logger setup -----------------------------------
        output_dir = Path(
            self.config.get("output", {}).get("output_dir", "brain_automl_output")
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        log_level = self.config.get("logging", {}).get("level", "INFO")
        logger = kwargs.pop("run_logger", None)
        if logger is None:
            logger = setup_run_logger(output_dir, level=log_level)

        # --- Data preparation -------------------------------------------------
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

        logger.info(
            f"Item: {selected_item} | Freq: {inferred_frequency} | "
            f"Horizon: {len(test_df)} | Train rows: {len(train_df)} | "
            f"Backends: {selected_backends}"
        )

        prediction_frame = test_df[["ds", "y"]].rename(columns={"y": "actual"}).copy()
        results: List[ModalityResult] = []
        metric_rows: List[Dict[str, Any]] = []

        # --- Backend loop with progress bar -----------------------------------
        for backend_name in tqdm(selected_backends, desc="Backends", unit="backend", leave=True):
            backend_output_dir = str(output_dir / backend_name)
            Path(backend_output_dir).mkdir(parents=True, exist_ok=True)

            backend_cls = BACKEND_REGISTRY.get(backend_name)
            backend = backend_cls()
            logger.info(f"[START] {backend_name}")
            t0 = time.perf_counter()
            try:
                model = backend.fit(
                    train_df,
                    None,
                    task="forecasting",
                    prediction_length=len(test_df),
                    horizon=len(test_df),
                    frequency=inferred_frequency,
                    seasonality=seasonality,
                    output_dir=backend_output_dir,
                    run_logger=logger,
                    **kwargs,
                )
                model_info = self._extract_backend_model_info(backend_name, model)
                raw_prediction = backend.predict(
                    model,
                    test_df,
                    task="forecasting",
                    prediction_length=len(test_df),
                    horizon=len(test_df),
                    frequency=inferred_frequency,
                    seasonality=seasonality,
                    output_dir=backend_output_dir,
                    run_logger=logger,
                    **kwargs,
                )
                normalized_prediction = normalize_prediction_frame(
                    raw_prediction,
                    backend_name=backend_name,
                    expected_dates=test_df["ds"],
                )
                prediction_frame[backend_name] = normalized_prediction["prediction"].to_numpy()

                # ── AutoGluon: extract per-internal-model predictions ──────
                if backend_name == "autogluon_timeseries":
                    predictor = model.get("predictor")
                    train_data_ag = model.get("train_data")
                    if predictor is not None and train_data_ag is not None:
                        try:
                            ag_lb = predictor.leaderboard(silent=True)
                            internal_models = ag_lb["model"].tolist() if "model" in ag_lb.columns else []
                            for ag_model_name in internal_models[:10]:  # top-10 max
                                if "WeightedEnsemble" in str(ag_model_name):
                                    continue
                                try:
                                    ag_fcst = predictor.predict(train_data_ag, model=ag_model_name)
                                    ag_df = ag_fcst.reset_index()
                                    ag_df = ag_df.rename(columns={"timestamp": "ds", "mean": "prediction"})
                                    ag_df["ds"] = pd.to_datetime(ag_df["ds"])
                                    norm_ag = normalize_prediction_frame(
                                        ag_df[["ds", "prediction"]],
                                        backend_name=ag_model_name,
                                        expected_dates=test_df["ds"],
                                    )
                                    col_name = f"ag_{ag_model_name}"
                                    prediction_frame[col_name] = norm_ag["prediction"].to_numpy()
                                    ag_metrics = compute_forecast_metrics(
                                        y_train=train_df["y"],
                                        y_true=test_df["y"],
                                        y_pred=norm_ag["prediction"],
                                        seasonality=seasonality,
                                    )
                                    metric_rows.append({
                                        "backend": col_name,
                                        "selected_model": ag_model_name,
                                        "trained_models": ag_model_name,
                                        "autogluon_internal": True,
                                        **ag_metrics,
                                    })
                                except Exception:
                                    pass
                        except Exception:
                            pass
                metrics = compute_forecast_metrics(
                    y_train=train_df["y"],
                    y_true=test_df["y"],
                    y_pred=normalized_prediction["prediction"],
                    seasonality=seasonality,
                )
                elapsed = time.perf_counter() - t0
                rmse = metrics.get("rmse", float("nan"))
                mae = metrics.get("mae", float("nan"))
                logger.info(
                    f"[ OK ] {backend_name} — {elapsed:.1f}s | "
                    f"RMSE={rmse:.4f}  MAE={mae:.4f}"
                )
                results.append(
                    ModalityResult(
                        modality=self.modality,
                        backend=backend_name,
                        task="forecasting",
                        predictions=normalized_prediction,
                        metrics=metrics,
                        fitted_model=model,
                        metadata={
                            "status": "ok",
                            "item_id": selected_item,
                            "frequency": inferred_frequency,
                            "horizon": len(test_df),
                            "selected_model": model_info.get("selected_model"),
                            "trained_models": model_info.get("trained_models"),
                            "elapsed_seconds": round(elapsed, 2),
                        },
                    )
                )
                metric_rows.append(
                    {
                        "backend": backend_name,
                        "selected_model": model_info.get("selected_model"),
                        "trained_models": " | ".join(model_info.get("trained_models", [])),
                        **metrics,
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error(f"[FAIL] {backend_name} — {elapsed:.1f}s | {exc}")
                logger.debug(f"Full traceback for {backend_name}:", exc_info=True)
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
                            "elapsed_seconds": round(elapsed, 2),
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
            best = metrics_df.iloc[0]
            logger.info(
                f"Best backend: {best['backend']} | "
                f"RMSE={best.get('rmse', float('nan')):.4f}  "
                f"MAE={best.get('mae', float('nan')):.4f}"
            )

        comprehensive_results = pd.DataFrame()
        if include_comprehensive_experiments:
            logger.info("Running comprehensive decomposition/model experiment sweep")
            # Filter to current item_id if specified to avoid processing all stocks
            comp_data = data
            if item_id and item_id_column and item_id_column in data.columns:
                comp_data = data[data[item_id_column] == item_id].copy()
                logger.info(f"Comprehensive sweep filtered to item_id={item_id!r} | rows={len(comp_data)}")
            
            split_map = split_data_by_stock(
                dataframe=comp_data,
                stock_col=item_id_column or "Stock",
                date_col=timestamp_column,
                train_ratio=None,  # Ignore ratio when horizon is provided
                horizon=horizon,   # Use horizon-based split for consistency
            )
            if split_map:
                # Import lazily to avoid forcing heavy experiment dependencies
                # during standard backend-only forecasting runs.
                from brain_automl.experiments import run_comprehensive_experiments

                comprehensive_results = run_comprehensive_experiments(
                    data_by_stock=split_map,
                    forecast_columns=tuple(comprehensive_forecast_columns or (target_column,)),
                    decomposition_types=tuple(
                        comprehensive_decomposition_types
                        or ("original", "decomposition+model", "hybrid")
                    ),
                    model_types=tuple(
                        comprehensive_model_types
                        or ("ARIMA", "ExpSmoothing", "LSTM", "XGBoost", "RandomForest")
                    ),
                )
                logger.info(
                    "Comprehensive sweep complete | runs=%s",
                    len(comprehensive_results),
                )
            else:
                logger.warning(
                    "Comprehensive sweep skipped: no valid per-stock train/test splits"
                )

        logger.info("forecast_last_horizon complete")
        return {
            "item_id": selected_item,
            "frequency": inferred_frequency,
            "horizon": len(test_df),
            "train_data": train_df,
            "test_data": test_df,
            "predictions": prediction_frame,
            "metrics": metrics_df,
            "comprehensive_experiments": comprehensive_results,
            "results": results,
            "output_dir": str(output_dir),
        }

    def backtest(
        self,
        series_data: pd.DataFrame,
        forecast_fn: Any,
        horizon: int = 30,
        n_windows: int = 3,
        min_train_size: int = 120,
        seasonality: int = DEFAULT_BUSINESS_SEASONALITY,
    ) -> Dict[str, Any]:
        """Run expanding-window backtesting for a standardized time-series frame."""
        return expanding_window_backtest(
            series_data=series_data,
            forecast_fn=forecast_fn,
            horizon=horizon,
            n_windows=n_windows,
            min_train_size=min_train_size,
            seasonality=seasonality,
        )

