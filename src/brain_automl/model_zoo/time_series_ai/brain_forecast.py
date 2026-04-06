"""High-level BrainAutoMLForecast — the single entry-point for time-series forecasting.

Consolidates all forecasting approaches from both experiment notebooks into one
clean API::

    from brain_automl.model_zoo.time_series_ai import BrainAutoMLForecast

    model = BrainAutoMLForecast(horizon=30, target_column="Close")
    model.fit(df)
    model.leaderboard()
    model.plot_forecast()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from brain_automl.config import get_default_config
from brain_automl.data_processing import split_data_by_stock
from brain_automl.model_zoo.time_series_ai.backtesting import expanding_window_backtest
from brain_automl.model_zoo.time_series_ai.data_preparation import (
    DEFAULT_BUSINESS_SEASONALITY,
    compute_forecast_metrics,
    to_standard_timeseries_format,
    select_item_series,
)
from brain_automl.model_zoo.time_series_ai.time_series_executor import TimeSeriesAutoML
from brain_automl.utilities.plotting import plot_forecast_vs_actual, plot_metric_bars
from brain_automl.utilities.run_logging import setup_run_logger


class BrainAutoMLForecast:
    """Unified AutoML forecasting interface.

    Parameters
    ----------
    horizon : int
        Number of future time steps to forecast (default 30).
    timestamp_column : str
        Name of the date/timestamp column in the input DataFrame.
    target_column : str
        Name of the target column to forecast.
    item_id_column : str or None
        Column that identifies each time series (e.g. "Stock"). When *None* the
        entire dataset is treated as a single series.
    include_backends : bool
        Run AutoGluon / StatsForecast / FLAML / Chronos backends (default True).
    include_comprehensive : bool
        Run the classical decomposition × model sweep (ARIMA, LSTM, XGBoost …).
        Warning: can be slow for large datasets (default False).
    include_advanced_dl : bool
        Run advanced deep-learning models (BiLSTM, Transformer, AttentionForecaster,
        Hybrid_ARIMA_BiLSTM_XGB). Requires TensorFlow (default False).
    backends : iterable of str or None
        Override which backends to use. ``None`` uses the config defaults.
    frequency : str
        Pandas frequency alias for the time series (default ``"B"`` = business days).
    seasonality : int
        Seasonal period for statistical models (default 5 = weekly).
    config : dict or None
        brain_automl configuration override. ``None`` uses :func:`get_default_config`.
    output_dir : str or Path or None
        Directory for AutoGluon / log artefacts.
    """

    def __init__(
        self,
        horizon: int = 30,
        timestamp_column: str = "Date",
        target_column: str = "Close",
        item_id_column: Optional[str] = None,
        include_backends: bool = True,
        include_comprehensive: bool = False,
        include_advanced_dl: bool = False,
        backends: Optional[Iterable[str]] = None,
        frequency: str = "B",
        seasonality: int = DEFAULT_BUSINESS_SEASONALITY,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.horizon = horizon
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.item_id_column = item_id_column
        self.include_backends = include_backends
        self.include_comprehensive = include_comprehensive
        self.include_advanced_dl = include_advanced_dl
        self.backends = list(backends) if backends is not None else None
        self.frequency = frequency
        self.seasonality = seasonality
        self.config = config or get_default_config()
        self.output_dir = Path(
            output_dir
            or self.config.get("output", {}).get("output_dir", "brain_automl_output")
        ).resolve()

        # Populated by fit()
        self._fit_result: Optional[Dict[str, Any]] = None
        self._leaderboard_df: Optional[pd.DataFrame] = None
        self._comprehensive_df: Optional[pd.DataFrame] = None
        self._advanced_df: Optional[pd.DataFrame] = None
        self._data: Optional[pd.DataFrame] = None
        self._multi_leaderboard_df: Optional[pd.DataFrame] = None  # populated by fit_all_stocks()
        self._stock_predictions: Dict[str, pd.DataFrame] = {}       # {stock: pred_df} with ds, actual, model cols
        self._stock_train_tail: Dict[str, pd.DataFrame] = {}        # {stock: last N rows of train for context}

    # ------------------------------------------------------------------ fit --

    def fit(
        self,
        data: pd.DataFrame,
        item_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "BrainAutoMLForecast":
        """Fit all configured forecasting approaches to *data*.

        Parameters
        ----------
        data : pd.DataFrame
            Raw time-series data (with ``timestamp_column`` and ``target_column``).
        item_id : str or None
            Specific item/stock to select. When *None* the first item is used.
        **kwargs
            Forwarded to :meth:`TimeSeriesAutoML.forecast_last_horizon`.

        Returns
        -------
        self  (for method chaining)
        """
        self._data = data.copy()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_run_logger(self.output_dir)

        logger.info("─" * 60)
        logger.info("BrainAutoMLForecast.fit() starting")
        logger.info(f"  Data shape       : {data.shape}")
        logger.info(f"  Target           : {self.target_column}")
        logger.info(f"  Timestamp col    : {self.timestamp_column}")
        logger.info(f"  Item-ID col      : {self.item_id_column!r}")
        logger.info(f"  Horizon          : {self.horizon}")
        logger.info(f"  include_backends : {self.include_backends}")
        logger.info(f"  include_comprehensive: {self.include_comprehensive}")
        logger.info(f"  include_advanced_dl : {self.include_advanced_dl}")
        logger.info("─" * 60)

        # ── 1. Backend forecasting ────────────────────────────────────────
        if self.include_backends:
            logger.info("[Phase 1/3] Running backend forecasting (AutoGluon / StatsForecast / FLAML …)")
            executor = TimeSeriesAutoML(config=self.config)
            self._fit_result = executor.forecast_last_horizon(
                data=data,
                timestamp_column=self.timestamp_column,
                target_column=self.target_column,
                item_id_column=self.item_id_column,
                item_id=item_id,
                horizon=self.horizon,
                backends=self.backends,
                frequency=self.frequency,
                seasonality=self.seasonality,
                include_comprehensive_experiments=self.include_comprehensive,
                comprehensive_forecast_columns=(self.target_column,),
                run_logger=logger,
                **kwargs,
            )
            self._comprehensive_df = self._fit_result.get("comprehensive_experiments", pd.DataFrame())
            logger.info(f"[Phase 1/3] Backend forecasting complete. "
                        f"Backends run: {[getattr(r, 'backend', r.get('backend', '?') if isinstance(r, dict) else '?') for r in self._fit_result.get('results', [])]}")
        else:
            # Build minimal result structure without running backends
            standard_df = to_standard_timeseries_format(
                data,
                target_column=self.target_column,
                timestamp_column=self.timestamp_column,
                item_id_column=self.item_id_column,
            )
            series_df, selected_item = select_item_series(standard_df, item_id=item_id)
            n = len(series_df)
            if n <= self.horizon:
                raise ValueError("Not enough data for the requested horizon")
            train_df = series_df.iloc[: n - self.horizon].copy()
            test_df = series_df.iloc[n - self.horizon :].copy()
            self._fit_result = {
                "item_id": selected_item,
                "train_data": train_df,
                "test_data": test_df,
                "predictions": test_df[["ds"]].rename(columns={"ds": "ds"}).assign(actual=test_df["y"].values),
                "metrics": pd.DataFrame(),
                "comprehensive_experiments": pd.DataFrame(),
                "results": [],
            }
            self._comprehensive_df = pd.DataFrame()

        # ── 2. Optional classical comprehensive experiments ────────────────
        if self.include_comprehensive and (self._comprehensive_df is None or self._comprehensive_df.empty):
            logger.info("[Phase 2/3] Running comprehensive decomposition/model experiment sweep")
            try:
                from brain_automl.experiments import run_comprehensive_experiments

                split_map = split_data_by_stock(
                    dataframe=data,
                    stock_col=self.item_id_column or "Stock",
                    date_col=self.timestamp_column,
                )
                if split_map:
                    self._comprehensive_df = run_comprehensive_experiments(
                        data_by_stock=split_map,
                        forecast_columns=(self.target_column,),
                    )
                    logger.info(f"[Phase 2/3] Comprehensive sweep done. Rows: {len(self._comprehensive_df)}")
            except Exception as exc:
                logger.warning(f"[Phase 2/3] Comprehensive experiments failed: {exc}", exc_info=True)

        # ── 3. Optional advanced DL experiments ───────────────────────────
        if self.include_advanced_dl:
            logger.info("[Phase 3/3] Running advanced DL experiments (BiLSTM / Transformer / Attention / HybridStack)")
            try:
                from brain_automl.experiments import run_advanced_experiments

                split_map = split_data_by_stock(
                    dataframe=data,
                    stock_col=self.item_id_column or "Stock",
                    date_col=self.timestamp_column,
                )
                if split_map:
                    self._advanced_df = run_advanced_experiments(
                        data_by_stock=split_map,
                        targets=(self.target_column,)
                    )
                    logger.info(f"[Phase 3/3] Advanced DL done. Rows: {len(self._advanced_df)}")
            except Exception as exc:
                logger.warning(f"[Phase 3/3] Advanced DL experiments failed: {exc}", exc_info=True)

        # ── 4. Build unified leaderboard ──────────────────────────────────
        logger.info("[Leaderboard] Building unified leaderboard …")
        self._leaderboard_df = self._build_leaderboard()
        logger.info("fit() complete — leaderboard rows: %d", len(self._leaderboard_df))
        return self

    # --------------------------------------------------------- leaderboard --

    def leaderboard(self) -> pd.DataFrame:
        """Return a unified leaderboard DataFrame sorted by RMSE (ascending).

        Columns include ``model``, ``library``, ``rmse``, ``mae``, ``mape``,
        ``mase``, ``directional_accuracy``, and others where available.
        """
        if self._leaderboard_df is None:
            raise RuntimeError("Call fit() before leaderboard()")
        return self._leaderboard_df.copy()

    # ----------------------------------------------------------- predictions --

    def predictions(self) -> pd.DataFrame:
        """Return the backend prediction frame with columns ``ds``, ``actual``,
        and one column per backend.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() before predictions()")
        return self._fit_result["predictions"].copy()

    # -------------------------------------------------------------- backtest --

    def backtest(
        self,
        forecast_fn: Any = None,
        n_windows: int = 3,
        min_train_size: int = 120,
    ) -> Dict[str, Any]:
        """Run expanding-window backtesting.

        Parameters
        ----------
        forecast_fn : callable or None
            ``forecast_fn(train_df, horizon) -> {"predictions": array}``.
            When *None*, a naive last-value forecast is used.
        n_windows : int
            Number of non-overlapping back-test windows (default 3).
        min_train_size : int
            Minimum rows required for a window to be valid (default 120).

        Returns
        -------
        dict with keys ``per_window`` (DataFrame) and ``aggregate`` (dict).
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() before backtest()")

        if forecast_fn is None:
            def forecast_fn(train_df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
                last_val = float(train_df["y"].iloc[-1])
                return {"predictions": [last_val] * horizon}

        item_id = self._fit_result.get("item_id")
        standard_df = to_standard_timeseries_format(
            self._data,
            target_column=self.target_column,
            timestamp_column=self.timestamp_column,
            item_id_column=self.item_id_column,
        )
        series_df, _ = select_item_series(standard_df, item_id=item_id)

        return expanding_window_backtest(
            series_data=series_df,
            forecast_fn=forecast_fn,
            horizon=self.horizon,
            n_windows=n_windows,
            min_train_size=min_train_size,
            seasonality=self.seasonality,
        )

    # --------------------------------------------------------- plot_forecast --

    def plot_forecast(self, backend: Optional[str] = None):
        """Return a Plotly figure of forecast vs actual.

        Parameters
        ----------
        backend : str or None
            Which backend column to plot. Defaults to the best (lowest RMSE) backend.
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() before plot_forecast()")

        pred_df = self._fit_result["predictions"]
        backend_cols = [c for c in pred_df.columns if c not in {"ds", "actual"}]

        if not backend_cols:
            raise ValueError("No backend predictions available in fit result")

        if backend is None:
            lb = self._leaderboard_df
            if lb is not None and not lb.empty and "model" in lb.columns:
                top_model = str(lb.iloc[0]["model"])
                backend = top_model if top_model in backend_cols else backend_cols[0]
            else:
                backend = backend_cols[0]

        train_df = self._fit_result.get("train_data")
        train_dates = pd.to_datetime(train_df["ds"]) if train_df is not None else None
        train_values = train_df["y"].values if train_df is not None else None

        return plot_forecast_vs_actual(
            dates=pd.to_datetime(pred_df["ds"]),
            actual=pred_df["actual"],
            forecasts={backend: pred_df[backend]},
            title=f"Forecast vs Actual — {self.target_column} ({backend})",
            train_dates=train_dates,
            train_values=train_values,
        )

    # ---------------------------------------------------------- plot_metrics --

    def plot_metrics(self, metrics: Sequence[str] = ("rmse", "mae", "mape")):
        """Return a Plotly bar-chart comparing all models by specified metrics."""
        lb = self.leaderboard()
        return plot_metric_bars(
            lb,
            metrics_to_plot=list(metrics),
            group_col="model",
            title=f"Model Comparison — {self.target_column}",
        )

    # ---------------------------------------------------------------- helpers --

    def _build_leaderboard(self) -> pd.DataFrame:
        """Combine backend metrics, comprehensive sweep results, and advanced DL into one frame."""
        frames: List[pd.DataFrame] = []

        # Backend metrics
        if self._fit_result:
            bk_metrics = self._fit_result.get("metrics", pd.DataFrame())
            if isinstance(bk_metrics, pd.DataFrame) and not bk_metrics.empty:
                bk = bk_metrics.copy()
                bk["library"] = "backend"
                bk = bk.rename(columns={"backend": "model"}) if "backend" in bk.columns else bk
                frames.append(bk)

        # Comprehensive decomposition/model experiments
        if isinstance(self._comprehensive_df, pd.DataFrame) and not self._comprehensive_df.empty:
            comp = self._comprehensive_df.copy()
            comp["model"] = comp.get("Decomposition", "") + "/" + comp.get("Model", "")
            comp["library"] = "classical"
            comp = comp.rename(columns={
                "MAE": "mae", "RMSE": "rmse", "MAPE": "mape", "DA": "directional_accuracy",
            })
            frames.append(comp)

        # Advanced DL experiments
        if isinstance(self._advanced_df, pd.DataFrame) and not self._advanced_df.empty:
            adv = self._advanced_df.copy()
            adv["model"] = adv["Model"] + "_" + adv.get("Setup", "")
            adv["library"] = "advanced_dl"
            adv = adv.rename(columns={
                "MAE": "mae", "RMSE": "rmse", "MAPE": "mape", "DA": "directional_accuracy",
            })
            frames.append(adv)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # Ensure lowercase metric columns
        col_remap = {c: c.lower() for c in combined.columns if c.upper() in {"MAE", "RMSE", "MAPE", "DA"}}
        combined = combined.rename(columns=col_remap)

        # Sort by RMSE
        if "rmse" in combined.columns:
            combined = combined.sort_values("rmse", ascending=True).reset_index(drop=True)

        return combined

    # ------------------------------------------------------------------ repr --

    def __repr__(self) -> str:
        fitted = "fitted" if self._fit_result is not None else "unfitted"
        return (
            f"BrainAutoMLForecast("
            f"target={self.target_column!r}, "
            f"horizon={self.horizon}, "
            f"backends={self.include_backends}, "
            f"comprehensive={self.include_comprehensive}, "
            f"advanced_dl={self.include_advanced_dl}, "
            f"status={fitted})"
        )

    # ------------------------------------------------- fit_all_stocks --------

    def fit_all_stocks(
        self,
        data: pd.DataFrame,
        horizons: Sequence[int] = (7, 15, 30),
        **kwargs: Any,
    ) -> "BrainAutoMLForecast":
        """Train on every stock and evaluate at multiple horizons from one model.

        All models are trained once using the last ``max(horizons)`` days as the
        holdout window. Metrics are then computed for each horizon by slicing
        predictions — so no retraining is needed for shorter horizons.

        Results are stored in ``_multi_leaderboard_df`` and accessible via
        ``multi_stock_leaderboard()``.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset containing all stocks.
        horizons : sequence of int
            Horizon lengths to evaluate (default ``[7, 15, 30]``).
            Trains with ``max(horizons)`` as holdout; shorter horizons reuse the
            same predictions.
        """
        self._data = data.copy()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_run_logger(self.output_dir)

        horizons_sorted: List[int] = sorted(set(int(h) for h in horizons))
        max_horizon = max(horizons_sorted)

        logger.info("=" * 60)
        logger.info(f"fit_all_stocks() starting")
        logger.info(f"  Data shape  : {data.shape}")
        logger.info(f"  Horizons    : {horizons_sorted}  (max={max_horizon})")
        logger.info(f"  Target      : {self.target_column}")
        logger.info(f"  Item-ID col : {self.item_id_column!r}")
        logger.info("=" * 60)

        # Collect all unique stocks
        if self.item_id_column and self.item_id_column in data.columns:
            all_stocks: List[str] = sorted(data[self.item_id_column].dropna().unique().tolist())
        else:
            all_stocks = ["series_0"]

        logger.info(f"Stocks to process: {all_stocks} ({len(all_stocks)} total)")

        executor = TimeSeriesAutoML(config=self.config)
        rows: List[Dict[str, Any]] = []
        last_fit_result: Optional[Dict[str, Any]] = None

        for stock_idx, stock in enumerate(all_stocks, 1):
            logger.info(f"\n{'─'*50}")
            logger.info(f"[{stock_idx}/{len(all_stocks)}] Stock: {stock}")

            # ── Backend forecasting per stock ────────────────────────────
            try:
                fit_result = executor.forecast_last_horizon(
                    data=data,
                    timestamp_column=self.timestamp_column,
                    target_column=self.target_column,
                    item_id_column=self.item_id_column,
                    item_id=stock,
                    horizon=max_horizon,
                    backends=self.backends,
                    frequency=self.frequency,
                    seasonality=self.seasonality,
                    # Comprehensive experiments are executed explicitly below once per stock.
                    # Keep this disabled here to avoid duplicate runs.
                    include_comprehensive_experiments=False,
                    comprehensive_forecast_columns=(self.target_column,),
                    run_logger=logger,
                    **kwargs,
                )
                last_fit_result = fit_result
            except Exception as exc:
                logger.warning(f"[{stock}] Backend forecasting failed: {exc}", exc_info=True)
                continue

            pred_df = fit_result["predictions"]  # ds, actual, backend_col, ...
            train_df = fit_result["train_data"]
            backend_cols = [c for c in pred_df.columns if c not in {"ds", "actual"}]

            # Build a mapping from prediction column -> (library, model_name)
            col_map: Dict[str, Dict[str, str]] = {}
            for r in fit_result.get("results", []) or []:
                try:
                    backend_name = getattr(r, "backend", None) or r.metadata.get("backend")
                except Exception:
                    backend_name = None
                if not backend_name:
                    continue
                sel = None
                trained = None
                try:
                    sel = r.metadata.get("selected_model")
                    trained = r.metadata.get("trained_models")
                except Exception:
                    sel = None
                    trained = None
                # prediction_frame uses a column named after backend (e.g. 'statsforecast')
                col_map[str(backend_name)] = {
                    "library": str(backend_name),
                    "model": str(sel) if sel else str(backend_name),
                    "trained_models": " | ".join(trained) if isinstance(trained, (list, tuple)) else (str(trained) if trained else ""),
                }

            # Store per-stock predictions for visualization and persist to disk
            self._stock_predictions[stock] = pred_df.copy()
            self._stock_train_tail[stock] = train_df.tail(60).copy()
            try:
                preds_dir = self.output_dir / "predictions"
                preds_dir.mkdir(parents=True, exist_ok=True)
                pred_file = preds_dir / f"{stock}_predictions_h{max_horizon}.csv"
                pred_df.to_csv(pred_file, index=False)
                logger.info(f"Saved per-stock predictions to: {pred_file}")
            except Exception:
                logger.debug("Failed to persist per-stock predictions", exc_info=True)

            # ── Evaluate each model at each horizon ──────────────────────
            for col in backend_cols:
                # Determine library tag and model name using the column mapping
                is_ag_internal = col.startswith("ag_")
                if is_ag_internal:
                    library = "autogluon_internal"
                    model_name = col[3:]  # strip "ag_" prefix
                elif col in col_map:
                    library = col_map[col].get("library", "backend")
                    model_name = col_map[col].get("model", col)
                else:
                    library = "backend"
                    model_name = col

                for h in horizons_sorted:
                    actual_h = pred_df["actual"].iloc[:h].to_numpy(dtype=float)
                    pred_h = pred_df[col].iloc[:h].to_numpy(dtype=float)
                    if pd.isnull(pred_h).all():
                        logger.info(f"  [{stock}/{col}/h={h}] all-NaN — skipped")
                        continue
                    # fill any partial NaNs with forward-fill
                    import numpy as _np
                    mask = _np.isnan(pred_h)
                    if mask.any():
                        pred_h = pd.Series(pred_h).ffill().bfill().to_numpy(dtype=float)

                    try:
                        metrics = compute_forecast_metrics(
                            y_train=train_df["y"],
                            y_true=actual_h,
                            y_pred=pred_h,
                            seasonality=self.seasonality,
                        )
                        rows.append({
                            "stock": stock,
                            "horizon": h,
                            "model": model_name,
                            "library": library,
                            **metrics,
                        })
                    except Exception as exc:
                        logger.warning(f"  [{stock}/{col}/h={h}] metrics failed: {exc}")

            # ── Optional comprehensive sweep (uses split_data_by_stock for this stock) ─
            if self.include_comprehensive:
                try:
                    from brain_automl.experiments import run_comprehensive_experiments

                    stock_df = data[data[self.item_id_column] == stock].copy() if self.item_id_column else data
                    split_map = split_data_by_stock(
                        dataframe=stock_df,
                        stock_col=self.item_id_column or "Stock",
                        date_col=self.timestamp_column,
                        train_ratio=1.0 - max_horizon / len(stock_df),
                    )
                    if split_map:
                        comp_df = run_comprehensive_experiments(
                            data_by_stock=split_map,
                            forecast_columns=(self.target_column,),
                        )
                        for _, row in comp_df.iterrows():
                            for h in horizons_sorted:
                                rows.append({
                                    "stock": stock,
                                    "horizon": h,
                                    "model": f"{row.get('Decomposition','')}/{row.get('Model','')}",
                                    "library": "classical",
                                    "mae": row.get("MAE", float("nan")),
                                    "rmse": row.get("RMSE", float("nan")),
                                    "mape": row.get("MAPE", float("nan")),
                                    "column": row.get("Column", ""),
                                })
                except Exception as exc:
                    logger.warning(f"[{stock}] Comprehensive failed: {exc}", exc_info=True)

            # ── Optional advanced DL per stock ───────────────────────────
            if self.include_advanced_dl:
                try:
                    from brain_automl.experiments import run_advanced_experiments

                    stock_df = data[data[self.item_id_column] == stock].copy() if self.item_id_column else data
                    split_map = split_data_by_stock(
                        dataframe=stock_df,
                        stock_col=self.item_id_column or "Stock",
                        date_col=self.timestamp_column,
                        train_ratio=1.0 - max_horizon / len(stock_df),
                    )
                    if split_map:
                        adv_df = run_advanced_experiments(
                            data_by_stock=split_map,
                            targets=(self.target_column,)
                        )
                        for _, row in adv_df.iterrows():
                            setup = row.get("Setup", "")
                            for h in horizons_sorted:
                                rows.append({
                                    "stock": stock,
                                    "horizon": h,
                                    "model": f"{row.get('Model','')}",
                                    "library": f"dl_{setup}",
                                    "mae": row.get("MAE", float("nan")),
                                    "rmse": row.get("RMSE", float("nan")),
                                    "mape": row.get("MAPE", float("nan")),
                                    "target": row.get("Target", ""),
                                })
                except Exception as exc:
                    logger.warning(f"[{stock}] Advanced DL failed: {exc}", exc_info=True)

        # ── Build multi-stock leaderboard ────────────────────────────────
        self._multi_leaderboard_df = pd.DataFrame(rows)
        if not self._multi_leaderboard_df.empty and "rmse" in self._multi_leaderboard_df.columns:
            self._multi_leaderboard_df = (
                self._multi_leaderboard_df
                .sort_values(["horizon", "rmse"], ascending=[True, True])
                .reset_index(drop=True)
            )

        # Keep last stock's fit_result for plot_forecast() compatibility
        if last_fit_result is not None:
            self._fit_result = last_fit_result
            self._leaderboard_df = self._build_leaderboard()

        logger.info(f"\nfit_all_stocks() complete — {len(self._multi_leaderboard_df)} leaderboard rows "
                    f"across {len(all_stocks)} stocks")
        return self

    # ---------------------------------------------- multi_stock_leaderboard --

    def multi_stock_leaderboard(
        self,
        horizon: Optional[int] = None,
        top_n_per_stock: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the multi-stock, multi-horizon leaderboard.

        Parameters
        ----------
        horizon : int or None
            Filter to a specific horizon. Returns all horizons when *None*.
        top_n_per_stock : int or None
            Keep only the top-N models per stock+horizon group. Returns all when *None*.
        """
        if self._multi_leaderboard_df is None:
            raise RuntimeError("Call fit_all_stocks() before multi_stock_leaderboard()")
        df = self._multi_leaderboard_df.copy()
        if horizon is not None:
            df = df[df["horizon"] == horizon]
        if top_n_per_stock is not None and "rmse" in df.columns:
            df = (
                df.sort_values("rmse")
                .groupby(["stock", "horizon"], group_keys=False)
                .head(top_n_per_stock)
                .reset_index(drop=True)
            )
        return df

    def best_model_per_stock(self, horizon: Optional[int] = None) -> pd.DataFrame:
        """Return the single best model (lowest RMSE) for each stock at each horizon."""
        if self._multi_leaderboard_df is None:
            raise RuntimeError("Call fit_all_stocks() before best_model_per_stock()")
        df = self._multi_leaderboard_df.copy()
        if horizon is not None:
            df = df[df["horizon"] == horizon]
        if "rmse" not in df.columns or df.empty:
            return df
        return (
            df.sort_values("rmse")
            .groupby(["stock", "horizon"], group_keys=False)
            .first()
            .reset_index()
        )

    def overall_best_models(self, top_n: int = 5) -> pd.DataFrame:
        """Return the top-N models ranked by mean RMSE across all stocks and horizons."""
        if self._multi_leaderboard_df is None:
            raise RuntimeError("Call fit_all_stocks() before overall_best_models()")
        df = self._multi_leaderboard_df.copy()
        if "rmse" not in df.columns or df.empty:
            return df
        return (
            df.groupby(["model", "library", "horizon"])["rmse"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_rmse", "std": "std_rmse", "count": "n_stocks"})
            .reset_index()
            .sort_values(["horizon", "mean_rmse"])
            .groupby("horizon", group_keys=False)
            .head(top_n)
            .reset_index(drop=True)
        )
