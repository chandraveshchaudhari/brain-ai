"""Forecasting metrics utilities for Brain-AI AutoML."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from brain_automl.model_zoo.time_series_ai.data_preparation import (
    DEFAULT_BUSINESS_SEASONALITY,
    compute_forecast_metrics,
)


EPS = 1e-10


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute directional accuracy using first differences sign matching."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) < 2 or len(y_pred_arr) < 2:
        return float("nan")

    true_direction = np.sign(np.diff(y_true_arr))
    pred_direction = np.sign(np.diff(y_pred_arr))
    return float(np.mean(true_direction == pred_direction))


def compute_forecasting_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute standard forecasting metrics used for leaderboard ranking."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))

    denom_mape = np.where(np.abs(y_true_arr) == 0, EPS, np.abs(y_true_arr))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom_mape)) * 100)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": directional_accuracy(y_true_arr, y_pred_arr),
    }


def compute_crps_from_quantiles(
    y_true: Sequence[float],
    quantile_forecasts: Dict[float, Sequence[float]],
    quantile_levels: Sequence[float],
) -> float:
    """Approximate CRPS from quantile forecasts with pinball loss."""
    y = np.asarray(y_true, dtype=float)
    levels = [float(q) for q in quantile_levels]
    if len(levels) < 2:
        return float("nan")

    total = 0.0
    for alpha in levels:
        q_vals = np.asarray(quantile_forecasts[alpha], dtype=float)
        err = y - q_vals
        total += np.mean(np.where(err >= 0, alpha * err, (alpha - 1.0) * err))
    return float(2.0 * total / len(levels))


def get_quantile_columns(pred_df: "np.typing.ArrayLike | object") -> Tuple[List[str], List[float]]:
    """Extract sorted quantile columns from a prediction dataframe-like object."""
    # We keep a soft dataframe contract here so notebooks can pass pandas frames directly.
    columns = list(getattr(pred_df, "columns", []))
    quantile_pairs: List[Tuple[float, str]] = []
    for col in columns:
        if col == "mean":
            continue
        try:
            q = float(col)
        except (TypeError, ValueError):
            continue
        quantile_pairs.append((q, str(col)))

    quantile_pairs.sort(key=lambda x: x[0])
    return [c for _, c in quantile_pairs], [q for q, _ in quantile_pairs]


def compute_full_metrics(
    y_train: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    quantile_forecasts: Optional[Dict[float, Sequence[float]]] = None,
    quantile_levels: Optional[Sequence[float]] = None,
    seasonality: int = DEFAULT_BUSINESS_SEASONALITY,
) -> Dict[str, float]:
    """Compute full forecast metrics suite including CRPS when quantiles exist."""
    metrics = compute_forecast_metrics(
        y_train=y_train,
        y_true=y_true,
        y_pred=y_pred,
        seasonality=seasonality,
    )
    metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred)

    if quantile_forecasts and quantile_levels:
        try:
            metrics["crps"] = compute_crps_from_quantiles(
                y_true=y_true,
                quantile_forecasts=quantile_forecasts,
                quantile_levels=quantile_levels,
            )
        except Exception:
            metrics["crps"] = float("nan")

    return metrics
