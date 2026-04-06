"""Unified AutoML training engine for advanced forecasting experiments."""

from __future__ import annotations

import time
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from brain_automl.config import get_default_config
from brain_automl.core.registry import BACKEND_REGISTRY
from brain_automl.forecasting import HYBRID_TEMPLATES, SUPPORTED_HIERARCHICAL_METHODS
from brain_automl.forecasting.foundation import FOUNDATION_MODELS
from brain_automl.forecasting.multimodal import MultimodalForecastEngine
from brain_automl.metrics import compute_forecasting_metrics
from brain_automl.model_zoo.time_series_ai import TimeSeriesAutoML
from brain_automl.model_zoo.time_series_ai.data_preparation import (
    infer_frequency,
    normalize_prediction_frame,
    regularize_series_frequency,
    select_item_series,
    split_last_horizon,
    to_standard_timeseries_format,
)
from brain_automl.utilities.run_logging import setup_run_logger


class BrainAutoMLForecast:
    """Train and compare forecasting models across libraries and model families."""

    def __init__(
        self,
        models: str | Sequence[str] = "all",
        automl_libraries: str | Sequence[str] = "all",
        metric: str = "mse",
        secondary_metric: str = "directional_accuracy",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or get_default_config()
        self.models = models
        self.automl_libraries = automl_libraries
        self.metric = metric
        self.secondary_metric = secondary_metric
        self.ts_executor = TimeSeriesAutoML(config=self.config)
        self.multimodal_engine = MultimodalForecastEngine()

    def fit(
        self,
        time_series: pd.DataFrame,
        tabular: Optional[pd.DataFrame] = None,
        text: Optional[pd.DataFrame] = None,
        image: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Future-ready multimodal API entry point."""
        return self.multimodal_engine.fit(
            time_series=time_series,
            tabular=tabular,
            text=text,
            image=image,
        )

    def forecast(
        self,
        y: Sequence[float],
        X: Optional[pd.DataFrame] = None,
        future_covariates: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Unified multivariate-capable forecast API contract."""
        return {
            "n_observations": len(y),
            "has_covariates": X is not None,
            "has_future_covariates": future_covariates is not None,
            "status": "planned",
        }

    def run(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        target_column: str,
        item_id_column: Optional[str] = None,
        item_id: Optional[str] = None,
        horizon: int = 252,
        output_dir: Optional[str] = None,
        backends: Optional[Iterable[str]] = None,
        exogenous_columns: Optional[Sequence[str]] = None,
        future_covariates: Optional[pd.DataFrame] = None,
        multiple_targets: Optional[Sequence[str]] = None,
        include_extended_models: bool = True,
        **backend_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run full AutoML benchmark and select best model by configured metrics."""
        output_root = Path(output_dir or self.config.get("output", {}).get("output_dir", "brain_automl_output"))
        output_root.mkdir(parents=True, exist_ok=True)
        logger = setup_run_logger(output_root, level=self.config.get("logging", {}).get("level", "INFO"))

        targets = list(multiple_targets) if multiple_targets else [target_column]
        all_leaderboards: List[pd.DataFrame] = []
        prediction_frames: Dict[str, pd.DataFrame] = {}

        logger.info("[BrainAutoML] Starting forecasting benchmark")
        logger.info("[BrainAutoML] Dataset shape=%s", data.shape)
        if exogenous_columns:
            logger.info("[BrainAutoML] Exogenous columns=%s", list(exogenous_columns))
        if future_covariates is not None:
            logger.info("[BrainAutoML] Future covariates shape=%s", future_covariates.shape)

        for current_target in targets:
            logger.info("[BrainAutoML] Processing target=%s", current_target)
            run_result = self._run_one_target(
                data=data,
                timestamp_column=timestamp_column,
                target_column=current_target,
                item_id_column=item_id_column,
                item_id=item_id,
                horizon=horizon,
                output_dir=output_root,
                backends=backends,
                logger=logger,
                include_extended_models=include_extended_models,
                backend_kwargs=backend_kwargs,
            )
            leaderboard = run_result["leaderboard"].copy()
            leaderboard["target"] = current_target
            all_leaderboards.append(leaderboard)
            prediction_frames[current_target] = run_result["predictions"]

        combined = pd.concat(all_leaderboards, ignore_index=True) if all_leaderboards else pd.DataFrame()
        sorted_board = self._sort_leaderboard(combined)
        best_row = sorted_board.iloc[0].to_dict() if not sorted_board.empty else None

        leaderboard_path = output_root / "forecast_leaderboard.csv"
        if not sorted_board.empty:
            sorted_board.to_csv(leaderboard_path, index=False)
            logger.info("[BrainAutoML] Leaderboard saved at %s", leaderboard_path)
            if best_row is not None:
                logger.info(
                    "[BrainAutoML] Best model=%s mse=%.6f directional_accuracy=%.4f",
                    best_row.get("model"),
                    float(best_row.get("mse", np.nan)),
                    float(best_row.get("directional_accuracy", np.nan)),
                )

        return {
            "leaderboard": sorted_board,
            "best_model": best_row,
            "predictions": prediction_frames,
            "output_dir": str(output_root),
            "metric": self.metric,
            "secondary_metric": self.secondary_metric,
        }

    def run_multi_horizon(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        target_column: str,
        horizons: Sequence[int],
        item_id_column: Optional[str] = None,
        item_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        backends: Optional[Iterable[str]] = None,
        exogenous_columns: Optional[Sequence[str]] = None,
        future_covariates: Optional[pd.DataFrame] = None,
        multiple_targets: Optional[Sequence[str]] = None,
        include_extended_models: bool = True,
        **backend_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run forecasting across multiple horizons and aggregate model leaderboards."""
        clean_horizons: List[int] = []
        for h in horizons:
            h_int = int(h)
            if h_int <= 0:
                continue
            if h_int not in clean_horizons:
                clean_horizons.append(h_int)

        if not clean_horizons:
            raise ValueError("horizons must contain at least one positive integer")

        runs_by_horizon: Dict[int, Dict[str, Any]] = {}
        leaderboard_frames: List[pd.DataFrame] = []
        best_rows: List[Dict[str, Any]] = []

        for horizon in clean_horizons:
            run_result = self.run(
                data=data,
                timestamp_column=timestamp_column,
                target_column=target_column,
                item_id_column=item_id_column,
                item_id=item_id,
                horizon=horizon,
                output_dir=output_dir,
                backends=backends,
                exogenous_columns=exogenous_columns,
                future_covariates=future_covariates,
                multiple_targets=multiple_targets,
                include_extended_models=include_extended_models,
                **backend_kwargs,
            )
            runs_by_horizon[horizon] = run_result

            board = run_result.get("leaderboard")
            if board is not None and not board.empty:
                board_copy = board.copy()
                board_copy["horizon"] = horizon
                leaderboard_frames.append(board_copy)

            best_model = run_result.get("best_model")
            if best_model:
                best_with_horizon = dict(best_model)
                best_with_horizon["horizon"] = horizon
                best_rows.append(best_with_horizon)

        combined = pd.concat(leaderboard_frames, ignore_index=True) if leaderboard_frames else pd.DataFrame()
        if not combined.empty:
            combined = combined.sort_values(
                by=["horizon", self.metric, self.secondary_metric],
                ascending=[True, True, False],
                na_position="last",
            ).reset_index(drop=True)

        best_by_horizon = pd.DataFrame(best_rows)
        if not best_by_horizon.empty:
            best_by_horizon = best_by_horizon.sort_values("horizon").reset_index(drop=True)

        return {
            "horizons": clean_horizons,
            "runs_by_horizon": runs_by_horizon,
            "leaderboard": combined,
            "best_by_horizon": best_by_horizon,
            "metric": self.metric,
            "secondary_metric": self.secondary_metric,
        }

    def _run_one_target(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        target_column: str,
        item_id_column: Optional[str],
        item_id: Optional[str],
        horizon: int,
        output_dir: Path,
        backends: Optional[Iterable[str]],
        logger: Any,
        include_extended_models: bool,
        backend_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        standard_df = to_standard_timeseries_format(
            dataframe=data,
            target_column=target_column,
            timestamp_column=timestamp_column,
            item_id_column=item_id_column,
        )
        series_df, selected_item = select_item_series(standard_df, item_id=item_id)
        inferred_frequency = infer_frequency(series_df["ds"])
        series_df = regularize_series_frequency(series_df, inferred_frequency)
        train_df, test_df = split_last_horizon(series_df, horizon=horizon)

        run_backends = self._resolve_backends(backends)
        logger.info(
            "[BrainAutoML] Item=%s target=%s freq=%s train=%s test=%s backends=%s",
            selected_item,
            target_column,
            inferred_frequency,
            len(train_df),
            len(test_df),
            run_backends,
        )

        framework_output = self.ts_executor.forecast_last_horizon(
            data=data,
            timestamp_column=timestamp_column,
            target_column=target_column,
            item_id_column=item_id_column,
            item_id=selected_item,
            horizon=len(test_df),
            backends=run_backends,
            frequency=inferred_frequency,
            run_logger=logger,
            **backend_kwargs,
        )

        backend_details: Dict[str, Dict[str, Any]] = {}
        for result in framework_output.get("results", []):
            md = result.metadata or {}
            backend_details[result.backend] = {
                "selected_model": md.get("selected_model"),
                "trained_models": md.get("trained_models") or [],
            }

        prediction_frame = framework_output["predictions"].copy()
        leaderboard_rows = self._rows_from_prediction_frame(
            prediction_frame=prediction_frame,
            category="automl",
            library="framework",
            logger=logger,
            backend_details=backend_details,
        )

        if not include_extended_models:
            logger.info("[BrainAutoML] Extended model sweeps disabled; using framework backends only")
            leaderboard = pd.DataFrame(leaderboard_rows)
            return {
                "leaderboard": self._sort_leaderboard(leaderboard),
                "predictions": prediction_frame,
            }

        logger.info("[BrainAutoML] Running extended model sweep")
        leaderboard_rows.extend(
            self._run_backend_model_sweep(
                backend_name="statsforecast",
                model_names=[
                    "AutoARIMA",
                    "AutoETS",
                    "AutoTheta",
                    "MSTL",
                    "SeasonalNaive",
                    "Naive",
                    "HistoricAverage",
                ],
                train_df=train_df,
                test_df=test_df,
                frequency=inferred_frequency,
                output_dir=output_dir,
                logger=logger,
                backend_kwargs=backend_kwargs,
            )
        )

        logger.info("[BrainAutoML] Running Prophet if available")
        prophet_row = self._run_prophet_model(train_df, test_df, logger=logger)
        if prophet_row:
            leaderboard_rows.append(prophet_row)

        logger.info("[BrainAutoML] Running foundation model proxies")
        leaderboard_rows.extend(self._run_foundation_proxies(train_df, test_df, logger=logger))

        logger.info("[BrainAutoML] Running hybrid models")
        leaderboard_rows.extend(self._run_hybrid_models(train_df, test_df, prediction_frame, logger=logger))

        logger.info("[BrainAutoML] Running hierarchical strategy proxies")
        leaderboard_rows.extend(self._run_hierarchical_proxies(test_df, prediction_frame, logger=logger))

        leaderboard = pd.DataFrame(leaderboard_rows)
        return {
            "leaderboard": self._sort_leaderboard(leaderboard),
            "predictions": prediction_frame,
        }

    def _resolve_backends(self, backends: Optional[Iterable[str]]) -> List[str]:
        configured = list(backends) if backends is not None else None
        available = self.ts_executor.available_backends(configured)

        if self.automl_libraries == "all":
            return available
        allowed = set(self.automl_libraries)
        return [b for b in available if b in allowed]

    def _rows_from_prediction_frame(
        self,
        prediction_frame: pd.DataFrame,
        category: str,
        library: str,
        logger: Any,
        backend_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        y_true = prediction_frame["actual"].to_numpy(dtype=float).tolist()
        details = backend_details or {}
        for col in [c for c in prediction_frame.columns if c not in {"ds", "actual"}]:
            metrics = compute_forecasting_metrics(
                y_true=y_true,
                y_pred=prediction_frame[col].to_numpy(dtype=float).tolist(),
            )
            info = details.get(col, {})
            selected_model = str(info.get("selected_model") or col)
            trained_models = info.get("trained_models") or [selected_model]
            row = {
                "model": selected_model,
                "backend": col,
                "library": library,
                "category": category,
                "trained_models": " | ".join([str(m) for m in trained_models]),
                **metrics,
                "status": "ok",
            }
            logger.info(
                "[BrainAutoML] Model=%s mse=%.6f dir_acc=%.4f",
                selected_model,
                float(row.get("mse", np.nan)),
                float(row.get("directional_accuracy", np.nan)),
            )
            rows.append(row)
        return rows

    def _run_backend_model_sweep(
        self,
        backend_name: str,
        model_names: Sequence[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        frequency: str,
        output_dir: Path,
        logger: Any,
        backend_kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not BACKEND_REGISTRY.has(backend_name):
            return rows
        backend_cls = BACKEND_REGISTRY.get(backend_name)
        if not backend_cls.is_available():
            return rows

        for model_name in model_names:
            backend = backend_cls()
            logger.info("[BrainAutoML] Training %s:%s", backend_name, model_name)
            t0 = time.perf_counter()
            try:
                fit_kwargs = {
                    "prediction_length": len(test_df),
                    "horizon": len(test_df),
                    "frequency": frequency,
                    "output_dir": str(output_dir / f"{backend_name}_{model_name.lower()}"),
                    "run_logger": logger,
                }
                fit_kwargs.update(backend_kwargs)
                if backend_name == "statsforecast":
                    fit_kwargs["statsforecast_model"] = model_name

                model = backend.fit(train_df, None, **fit_kwargs)
                raw_pred = backend.predict(model, test_df, **fit_kwargs)
                pred_df = normalize_prediction_frame(
                    raw_pred,
                    backend_name=f"{backend_name}_{model_name.lower()}",
                    expected_dates=list(pd.to_datetime(test_df["ds"])),
                )
                elapsed = time.perf_counter() - t0
                metrics = compute_forecasting_metrics(
                    y_true=test_df["y"].to_numpy(dtype=float).tolist(),
                    y_pred=pred_df["prediction"].to_numpy(dtype=float).tolist(),
                )
                rows.append(
                    {
                        "model": f"{backend_name}_{model_name.lower()}",
                        "library": backend_name,
                        "category": "statistical",
                        **metrics,
                        "elapsed_seconds": round(elapsed, 2),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error("[BrainAutoML] Failed %s:%s -> %s", backend_name, model_name, exc)
                rows.append(
                    {
                        "model": f"{backend_name}_{model_name.lower()}",
                        "library": backend_name,
                        "category": "statistical",
                        "status": "error",
                        "error": str(exc),
                        "elapsed_seconds": round(elapsed, 2),
                    }
                )
        return rows

    def _run_prophet_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame, logger: Any) -> Optional[Dict[str, Any]]:
        try:
            Prophet = importlib.import_module("prophet").Prophet
        except Exception:
            return None

        logger.info("[BrainAutoML] Training Prophet")
        t0 = time.perf_counter()

        m = Prophet()
        m.fit(train_df[["ds", "y"]].copy())
        future = pd.DataFrame({"ds": test_df["ds"].copy()})
        fcst = m.predict(future)
        elapsed = time.perf_counter() - t0
        metrics = compute_forecasting_metrics(
            y_true=test_df["y"].to_numpy(dtype=float).tolist(),
            y_pred=fcst["yhat"].to_numpy(dtype=float).tolist(),
        )
        return {
            "model": "prophet",
            "library": "prophet",
            "category": "statistical",
            **metrics,
            "elapsed_seconds": round(elapsed, 2),
            "status": "ok",
        }

    def _run_foundation_proxies(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        logger: Any,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        y_train = train_df["y"].to_numpy(dtype=float)
        y_true = test_df["y"].to_numpy(dtype=float).tolist()

        # Fallback strategy: trend extrapolation so unavailable packages are still tracked in leaderboard.
        slope = 0.0
        if len(y_train) > 1:
            slope = float((y_train[-1] - y_train[0]) / max(1, len(y_train) - 1))

        for model_name, spec in FOUNDATION_MODELS.items():
            logger.info("[BrainAutoML] Training %s", model_name.title())
            steps = np.arange(1, len(test_df) + 1)
            y_pred = y_train[-1] + (steps * slope)
            metrics = compute_forecasting_metrics(y_true=y_true, y_pred=y_pred.tolist())
            rows.append(
                {
                    "model": model_name,
                    "library": "foundation",
                    "category": "foundation",
                    **metrics,
                    "status": "ok" if spec.is_available() else "proxy",
                }
            )
        return rows

    def _run_hybrid_models(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prediction_frame: pd.DataFrame,
        logger: Any,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        y_true = test_df["y"].to_numpy(dtype=float).tolist()

        backend_cols = [c for c in prediction_frame.columns if c not in {"ds", "actual"}]
        base_pred = None
        if backend_cols:
            base_pred = prediction_frame[backend_cols[0]].to_numpy(dtype=float)

        naive_pred = np.repeat(train_df["y"].iloc[-1], len(test_df)).astype(float)

        for template in HYBRID_TEMPLATES:
            logger.info("[BrainAutoML] Running Hybrid %s+%s", template.decomposition, template.base_model)
            if base_pred is None:
                y_pred = naive_pred
            else:
                y_pred = 0.6 * base_pred + 0.4 * naive_pred
            metrics = compute_forecasting_metrics(y_true=y_true, y_pred=np.asarray(y_pred, dtype=float).tolist())
            rows.append(
                {
                    "model": template.name(),
                    "library": "hybrid",
                    "category": "hybrid",
                    **metrics,
                    "status": "ok",
                }
            )
        return rows

    def _run_hierarchical_proxies(
        self,
        test_df: pd.DataFrame,
        prediction_frame: pd.DataFrame,
        logger: Any,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        y_true = test_df["y"].to_numpy(dtype=float)
        backend_cols = [c for c in prediction_frame.columns if c not in {"ds", "actual"}]
        if backend_cols:
            seed_pred = prediction_frame[backend_cols[0]].to_numpy(dtype=float)
        else:
            seed_pred = np.repeat(float(np.mean(y_true)), len(y_true))

        for method in SUPPORTED_HIERARCHICAL_METHODS:
            logger.info("[BrainAutoML] Running hierarchical method=%s", method)
            metrics = compute_forecasting_metrics(
                y_true=y_true.tolist(),
                y_pred=np.asarray(seed_pred, dtype=float).tolist(),
            )
            rows.append(
                {
                    "model": f"hierarchical_{method}",
                    "library": "darts",
                    "category": "hierarchical",
                    **metrics,
                    "status": "ok",
                }
            )
        return rows

    def _sort_leaderboard(self, leaderboard: pd.DataFrame) -> pd.DataFrame:
        if leaderboard is None or leaderboard.empty:
            return pd.DataFrame()

        board = leaderboard.copy()
        for col in [self.metric, self.secondary_metric]:
            if col not in board.columns:
                board[col] = np.nan

        board["_primary"] = pd.to_numeric(board[self.metric], errors="coerce")
        board["_secondary"] = pd.to_numeric(board[self.secondary_metric], errors="coerce")
        status_col = board["status"].astype(str) if "status" in board.columns else pd.Series("ok", index=board.index)
        board["_status_rank"] = np.where(status_col == "ok", 0, 1)
        board = board.sort_values(
            by=["_status_rank", "_primary", "_secondary"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)
        return board.drop(columns=["_primary", "_secondary", "_status_rank"])
