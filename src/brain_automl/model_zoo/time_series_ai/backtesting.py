"""Backtesting helpers for time-series forecasting experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from brain_automl.model_zoo.time_series_ai.data_preparation import DEFAULT_BUSINESS_SEASONALITY


def expanding_window_backtest(
    series_data: pd.DataFrame,
    forecast_fn: Callable[[pd.DataFrame, int], Dict[str, Any]],
    horizon: int = 30,
    n_windows: int = 3,
    min_train_size: int = 120,
    seasonality: int = DEFAULT_BUSINESS_SEASONALITY,
) -> Dict[str, Any]:
    """Run non-overlapping expanding-window backtesting on one series.

    Parameters
    ----------
    series_data:
        DataFrame with standardized columns `unique_id`, `ds`, `y`.
    forecast_fn:
        Callable that takes (train_df, horizon) and returns a dictionary with
        `predictions` and optional quantile payload.
    """
    n_rows = len(series_data)
    window_ends = []
    for i in range(n_windows):
        end = n_rows - i * horizon
        if end - horizon >= min_train_size:
            window_ends.append(end)
    window_ends = sorted(window_ends)

    if not window_ends:
        raise ValueError(
            f"Insufficient data for backtest: n={n_rows}, horizon={horizon}, min_train={min_train_size}"
        )

    # Import lazily to avoid package initialization cycles while importing brain_automl.
    from brain_automl.metrics import compute_full_metrics

    rows = []
    for idx, end in enumerate(window_ends):
        train = series_data.iloc[: end - horizon].copy().reset_index(drop=True)
        test = series_data.iloc[end - horizon : end].copy().reset_index(drop=True)
        cur_horizon = len(test)
        if cur_horizon == 0:
            continue

        try:
            forecast = forecast_fn(train, cur_horizon)
            preds = np.asarray(forecast["predictions"][:cur_horizon], dtype=float)
            metrics = compute_full_metrics(
                y_train=train["y"].to_numpy(dtype=float).tolist(),
                y_true=test["y"].to_numpy(dtype=float).tolist(),
                y_pred=preds.tolist(),
                quantile_forecasts=forecast.get("quantile_forecasts"),
                quantile_levels=forecast.get("quantile_levels"),
                seasonality=seasonality,
            )
            row_payload: Dict[str, Any] = dict(metrics)
            row_payload.update(
                {
                    "window": idx,
                    "train_size": len(train),
                    "test_size": cur_horizon,
                    "test_start": str(pd.Timestamp(test["ds"].iloc[0]).date()),
                    "test_end": str(pd.Timestamp(test["ds"].iloc[-1]).date()),
                    "status": "ok",
                }
            )
            rows.append(row_payload)
        except Exception as exc:
            rows.append(
                {
                    "window": idx,
                    "train_size": len(train),
                    "test_size": cur_horizon,
                    "test_start": str(pd.Timestamp(test["ds"].iloc[0]).date()) if cur_horizon else "N/A",
                    "test_end": str(pd.Timestamp(test["ds"].iloc[-1]).date()) if cur_horizon else "N/A",
                    "status": f"error: {exc}",
                }
            )

    result_df = pd.DataFrame(rows)
    ok_df = result_df[result_df["status"] == "ok"]
    if ok_df.empty:
        return {"per_window": result_df, "aggregate": {"n_windows": 0}}

    skip_cols = {"window", "train_size", "test_size", "test_start", "test_end", "status"}
    metric_cols = [c for c in ok_df.columns if c not in skip_cols]
    aggregate = ok_df[metric_cols].mean(numeric_only=True).to_dict()
    aggregate["n_windows"] = int(len(ok_df))
    return {"per_window": result_df, "aggregate": aggregate}
