"""Wrangling utilities migrated from forecasting notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def prepare_stock_forecasting_dataframe(
    dataframe: pd.DataFrame,
    date_col: str = "Date",
    group_col: str = "Stock",
    target_col: str = "Close",
    volatility_col: str = "Volatility",
    last_n_days: int = 252,
) -> pd.DataFrame:
    """Normalize, trim, and enrich stock forecasting data for benchmark tasks."""
    df = dataframe.copy()
    df.columns = [c.strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([group_col, date_col]).reset_index(drop=True)

    if last_n_days > 0:
        df = df.groupby(group_col, group_keys=False).tail(last_n_days).reset_index(drop=True)

    computed_col = "_vol_computed"
    df[computed_col] = np.log(df[target_col] / df.groupby(group_col)[target_col].shift(1)) ** 2

    if volatility_col in df.columns:
        df[volatility_col] = pd.to_numeric(df[volatility_col], errors="coerce").fillna(df[computed_col])
    else:
        df[volatility_col] = df[computed_col]

    df = df.dropna(subset=[volatility_col]).drop(columns=[computed_col]).reset_index(drop=True)
    return df


def load_stock_forecasting_dataset(
    data_path: str | Path,
    date_col: str = "Date",
    group_col: str = "Stock",
    target_col: str = "Close",
    volatility_col: str = "Volatility",
    last_n_days: int = 252,
) -> pd.DataFrame:
    """Load CSV and apply standard stock forecasting wrangling."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw_df = pd.read_csv(path)
    return prepare_stock_forecasting_dataframe(
        raw_df,
        date_col=date_col,
        group_col=group_col,
        target_col=target_col,
        volatility_col=volatility_col,
        last_n_days=last_n_days,
    )


def split_data_by_stock(
    dataframe: pd.DataFrame,
    stock_col: str = "Stock",
    date_col: str = "Date",
    train_ratio: Optional[float] = 0.8,
    horizon: Optional[int] = None,
) -> Dict[str, Dict[str, pd.DataFrame | pd.Timestamp]]:
    """Create chronological train/test splits per stock ticker.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Full dataset with all stocks.
    stock_col : str
        Column name for stock/item identifier.
    date_col : str
        Column name for timestamp.
    train_ratio : float, optional
        Fraction of data to use for training (0.0 to 1.0).
        Only used if `horizon` is None. Default 0.8.
    horizon : int, optional
        If provided, use the last `horizon` rows as test set (ignores train_ratio).
        This ensures consistent evaluation with a fixed test window.
    
    Returns
    -------
    Dict mapping stock -> {full, train, test, split_date}
    """
    out: Dict[str, Dict[str, pd.DataFrame | pd.Timestamp]] = {}
    stocks: List[str] = sorted(dataframe[stock_col].dropna().unique().tolist())

    for stock in stocks:
        stock_df = dataframe[dataframe[stock_col] == stock].copy()
        stock_df[date_col] = pd.to_datetime(stock_df[date_col], errors="coerce")
        stock_df = stock_df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        if stock_df.empty:
            continue

        # Use horizon-based split if provided, otherwise ratio-based
        if horizon is not None and horizon > 0:
            # Ensure we have enough data for the horizon
            if len(stock_df) <= horizon:
                continue  # Skip stocks with insufficient data
            split_idx = len(stock_df) - horizon
        else:
            split_idx = int(len(stock_df) * train_ratio)
            split_idx = max(1, min(split_idx, len(stock_df) - 1))
            
        out[stock] = {
            "full": stock_df,
            "train": stock_df.iloc[:split_idx].copy(),
            "test": stock_df.iloc[split_idx:].copy(),
            "split_date": pd.Timestamp(stock_df.iloc[split_idx][date_col]),
        }

    return out
