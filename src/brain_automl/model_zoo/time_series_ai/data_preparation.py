"""Shared data preparation utilities for time-series backends."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_BUSINESS_SEASONALITY = 5


def to_standard_timeseries_format(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    item_id_column: Optional[str] = None,
    item_id_value: str = "series_0",
) -> pd.DataFrame:
    """Convert a generic DataFrame into the common schema used by backends."""
    df = dataframe.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    df = df.dropna(subset=[timestamp_column, target_column]).sort_values(timestamp_column)
    if item_id_column is None:
        df["unique_id"] = item_id_value
    else:
        df["unique_id"] = df[item_id_column].astype(str)
    df["ds"] = df[timestamp_column]
    df["y"] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=["y"])
    return df[["unique_id", "ds", "y"]].reset_index(drop=True)


def select_item_series(
    dataframe: pd.DataFrame,
    item_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Select one item/series from a standardized time-series frame."""
    if dataframe.empty:
        raise ValueError("Input time-series data is empty after preprocessing")

    available_items = dataframe["unique_id"].astype(str)
    selected_item = item_id or available_items.iloc[0]
    selected_frame = dataframe[available_items == selected_item].copy()
    if selected_frame.empty:
        raise ValueError(f"Item '{selected_item}' was not found in the dataset")
    return selected_frame.reset_index(drop=True), selected_item


def split_last_horizon(
    dataframe: pd.DataFrame,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one standardized series into train and test using the last horizon rows."""
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive")
    if len(dataframe) <= horizon:
        raise ValueError("Forecast horizon must be smaller than the number of observations")
    train_df = dataframe.iloc[:-horizon].copy().reset_index(drop=True)
    test_df = dataframe.iloc[-horizon:].copy().reset_index(drop=True)
    return train_df, test_df


def infer_frequency(timestamps: pd.Series) -> str:
    """Infer a pandas-compatible frequency string from timestamps."""
    inferred = pd.infer_freq(pd.DatetimeIndex(timestamps))
    if inferred:
        return inferred
    return "B"


def regularize_series_frequency(
    dataframe: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    """Reindex one standardized series to a regular frequency and fill gaps."""
    if dataframe.empty:
        return dataframe.copy()

    item_id = str(dataframe["unique_id"].iloc[0])
    working = dataframe[["ds", "y"]].copy().sort_values("ds").set_index("ds")
    full_index = pd.date_range(working.index.min(), working.index.max(), freq=frequency)
    regularized = working.reindex(full_index)
    regularized["y"] = regularized["y"].interpolate(method="time").ffill().bfill()
    regularized = regularized.reset_index().rename(columns={"index": "ds"})
    regularized["unique_id"] = item_id
    return regularized[["unique_id", "ds", "y"]]


def to_autogluon_timeseries_format(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    item_id_column: Optional[str] = None,
) -> pd.DataFrame:
    """Convert generic dataframe into AutoGluon time-series schema.

    AutoGluon expects columns: item_id, timestamp, target.
    """
    df = dataframe.copy()
    if item_id_column is None:
        df["item_id"] = "series_0"
    else:
        df["item_id"] = df[item_id_column]
    df["timestamp"] = pd.to_datetime(df[timestamp_column])
    df["target"] = df[target_column]
    return df[["item_id", "timestamp", "target"]]


def to_neuralforecast_format(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert standardized time-series frame into NeuralForecast schema."""
    df = dataframe.copy()
    return df[["unique_id", "ds", "y"]]


def to_pycaret_timeseries_format(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Convert generic dataframe to pycaret-compatible indexed time-series."""
    df = dataframe.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column)
    return df.set_index(timestamp_column)[[target_column]]


def normalize_prediction_frame(
    predictions: pd.DataFrame,
    backend_name: str,
    expected_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Normalize backend predictions into a common ds/prediction schema."""
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(f"Predictions from '{backend_name}' must be a pandas DataFrame")

    df = predictions.copy()
    if "timestamp" in df.columns and "ds" not in df.columns:
        df = df.rename(columns={"timestamp": "ds"})
    if "date" in df.columns and "ds" not in df.columns:
        df = df.rename(columns={"date": "ds"})
    if "mean" in df.columns and "prediction" not in df.columns:
        df = df.rename(columns={"mean": "prediction"})

    value_columns = [col for col in df.columns if col not in {"unique_id", "item_id", "ds", "timestamp", "date"}]
    if "prediction" not in df.columns:
        if not value_columns:
            raise ValueError(f"Predictions from '{backend_name}' do not contain a forecast column")
        df = df.rename(columns={value_columns[0]: "prediction"})

    if "ds" not in df.columns:
        df["ds"] = pd.to_datetime(list(expected_dates))
    else:
        df["ds"] = pd.to_datetime(df["ds"])

    normalized = df[["ds", "prediction"]].copy().sort_values("ds").reset_index(drop=True)
    normalized["ds"] = pd.to_datetime(normalized["ds"])
    if len(normalized) != len(expected_dates):
        expected_df = pd.DataFrame({"ds": pd.to_datetime(list(expected_dates))})
        normalized = expected_df.merge(normalized, on="ds", how="left")
    return normalized


def compute_forecast_metrics(
    y_train: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    seasonality: int = DEFAULT_BUSINESS_SEASONALITY,
) -> Dict[str, float]:
    """Compute common forecast error metrics."""
    y_train_arr = np.asarray(y_train, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    denom_mape = np.where(np.abs(y_true_arr) == 0, 1e-10, np.abs(y_true_arr))
    denom_smape = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
    denom_smape = np.where(denom_smape == 0, 1e-10, denom_smape)
    wape_denom = np.sum(np.abs(y_true_arr))
    wape_denom = 1e-10 if wape_denom == 0 else wape_denom

    mase = float("nan")
    if len(y_train_arr) > seasonality:
        naive_errors = np.abs(y_train_arr[seasonality:] - y_train_arr[:-seasonality])
        scale = float(np.mean(naive_errors))
        if scale != 0:
            mase = float(np.mean(np.abs(y_true_arr - y_pred_arr)) / scale)

    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "mse": float(mean_squared_error(y_true_arr, y_pred_arr)),
        "rmse": rmse,
        "mape": float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom_mape)) * 100),
        "smape": float(np.mean(np.abs(y_true_arr - y_pred_arr) / denom_smape) * 100),
        "wape": float(np.sum(np.abs(y_true_arr - y_pred_arr)) / wape_denom * 100),
        "mase": mase,
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }
