"""Shared data preparation utilities for time-series backends."""

from __future__ import annotations

from typing import Optional

import pandas as pd


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
