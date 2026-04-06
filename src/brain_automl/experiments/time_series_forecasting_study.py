"""Notebook-migrated time-series forecasting experiment utilities.

This module consolidates reusable code from:
- examples/framework_forecasting_test.ipynb
- examples/experiments_forecasting_comparison.ipynb
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from brain_automl.metrics import compute_full_metrics
from brain_automl.model_zoo.time_series_ai.data_preparation import regularize_series_frequency


def decompose_series(
    series: Sequence[float],
    method: str = "additive",
    period: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Decompose a time series into trend/seasonal/residual components."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 10:
        vals = series.to_numpy(dtype=float)
        return {
            "original": {
                "trend": vals,
                "seasonal": np.zeros_like(vals),
                "residual": np.zeros_like(vals),
            }
        }

    if period is None:
        period = 5

    out: Dict[str, Dict[str, np.ndarray]] = {}

    try:
        if len(series) >= 2 * period:
            try:
                comp = seasonal_decompose(series, model=method, period=period, extrapolate_trend="freq")
            except TypeError:
                comp = seasonal_decompose(series, model=method, period=period)
            out["classical"] = {
                "trend": np.asarray(comp.trend, dtype=float),
                "seasonal": np.asarray(comp.seasonal, dtype=float),
                "residual": np.asarray(comp.resid, dtype=float),
            }
    except Exception:
        pass

    try:
        seasonal_window = max(7, min(period * 2 + 1, len(series) // 2 * 2 + 1))
        if seasonal_window % 2 == 0:
            seasonal_window += 1
        stl = STL(series.reset_index(drop=True), seasonal=seasonal_window, robust=True)
        stl_out = stl.fit()
        out["stl"] = {
            "trend": np.asarray(stl_out.trend, dtype=float),
            "seasonal": np.asarray(stl_out.seasonal, dtype=float),
            "residual": np.asarray(stl_out.resid, dtype=float),
        }
    except Exception:
        pass

    vals = series.to_numpy(dtype=float)
    out["original"] = {
        "trend": vals,
        "seasonal": np.zeros_like(vals),
        "residual": np.zeros_like(vals),
    }
    return out


def create_lag_features(series: pd.Series, n_lags: int = 20) -> pd.DataFrame:
    """Create lagged feature matrix."""
    data = pd.DataFrame()
    for lag in range(1, n_lags + 1):
        data[f"lag_{lag}"] = series.shift(lag)
    return data.dropna()


def prepare_ml_data(
    train_series: pd.Series,
    test_series: pd.Series,
    n_lags: int = 20,
) -> Optional[Dict[str, np.ndarray]]:
    """Prepare lag-based train/test arrays for ML regressors."""
    try:
        x_train = create_lag_features(train_series, n_lags=n_lags)
        y_train = train_series.iloc[n_lags:].to_numpy(dtype=float)

        combined = pd.concat([train_series, test_series], ignore_index=True)
        x_test_all = create_lag_features(combined, n_lags=n_lags)
        x_test = x_test_all.iloc[len(train_series) :].reset_index(drop=True)
        y_test = (
            test_series.iloc[n_lags:].to_numpy(dtype=float)
            if len(test_series) > n_lags
            else test_series.to_numpy(dtype=float)
        )

        if len(x_train) == 0 or len(x_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
            return None

        return {
            "X_train": x_train.to_numpy(dtype=float),
            "y_train": y_train,
            "X_test": x_test.to_numpy(dtype=float)[: len(y_test)],
            "y_test": y_test,
        }
    except Exception:
        return None


def calculate_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Return MAE, RMSE, MAPE, DA metric bundle."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    da = float(np.mean((np.diff(y_true_arr) > 0) == (np.diff(y_pred_arr) > 0)) * 100) if len(y_true_arr) > 1 else float("nan")
    return {
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "MAPE": float(mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100),
        "DA": da,
    }


def train_arima_model(
    train_series: Sequence[float],
    test_series: Sequence[float],
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Optional[np.ndarray]:
    """Train ARIMA and forecast test horizon."""
    try:
        fitted = ARIMA(train_series, order=order).fit()
        return np.asarray(fitted.get_forecast(steps=len(test_series)).predicted_mean, dtype=float)
    except Exception:
        return None


def train_exponential_smoothing(
    train_series: Sequence[float],
    test_series: Sequence[float],
) -> Optional[np.ndarray]:
    """Train Exponential Smoothing and forecast test horizon."""
    try:
        fitted = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit()
        return np.asarray(fitted.forecast(steps=len(test_series)), dtype=float)
    except Exception:
        return None


def train_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    epochs: int = 150,
    verbose: int = 0,
) -> Optional[np.ndarray]:
    """Train LSTM regressor with lazy TensorFlow import."""
    try:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
    except Exception:
        return None

    try:
        if len(x_train) < 2 or len(x_test) < 1:
            return None

        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        x_test = np.asarray(x_test, dtype=float)
        if len(x_train) != len(y_train):
            return None

        x_train_lstm = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        model = Sequential(
            [
                LSTM(64, activation="relu", return_sequences=True, input_shape=(x_train_lstm.shape[1], x_train_lstm.shape[2])),
                Dropout(0.2),
                LSTM(32, activation="relu"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        model.fit(
            x_train_lstm,
            y_train,
            epochs=epochs,
            verbose=verbose,
            batch_size=min(16, len(x_train)),
            validation_split=0.1,
        )
        pred = model.predict(x_test_lstm, verbose=0).flatten()
        return np.asarray(pred, dtype=float)
    except Exception:
        return None


def train_xgboost_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> Optional[np.ndarray]:
    """Train XGBoost model with notebook hyperparameters."""
    try:
        import xgboost as xgb
    except Exception:
        return None

    try:
        if len(x_train) < 2 or len(x_test) < 1:
            return None

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float), verbose=False)
        return np.asarray(model.predict(np.asarray(x_test, dtype=float)), dtype=float)
    except Exception:
        return None


def train_random_forest_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> Optional[np.ndarray]:
    """Train RandomForest model with notebook hyperparameters."""
    try:
        if len(x_train) < 2 or len(x_test) < 1:
            return None
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float))
        return np.asarray(model.predict(np.asarray(x_test, dtype=float)), dtype=float)
    except Exception:
        return None


def train_hybrid_model(
    train_series: Sequence[float],
    test_series: Sequence[float],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> Optional[np.ndarray]:
    """Hybrid model: ARIMA trend + LSTM residual."""
    try:
        train_arr = np.asarray(train_series, dtype=float)
        test_arr = np.asarray(test_series, dtype=float)
        if len(train_arr) < 20 or len(x_train) < 2:
            return None

        period = min(60, max(2, len(train_arr) // 4))
        try:
            decomp = seasonal_decompose(pd.Series(train_arr), model="additive", period=period, extrapolate_trend="freq")
        except TypeError:
            decomp = seasonal_decompose(pd.Series(train_arr), model="additive", period=period)

        trend_pred = train_arima_model(np.asarray(decomp.trend, dtype=float), test_arr, order=(2, 1, 1))
        residual_pred = train_lstm_model(x_train, y_train, x_test, epochs=100, verbose=0)

        if trend_pred is None or residual_pred is None or len(trend_pred) != len(residual_pred):
            return None
        return 0.6 * trend_pred + 0.4 * residual_pred
    except Exception:
        return None


def build_feature_frame(df_in: pd.DataFrame, target_col: str, mode: str = "univariate") -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix for univariate vs multivariate advanced experiments."""
    df_local = df_in.copy()
    base_cols = ["Close", "High", "Low", "Volume", "Volatility"]

    if mode == "univariate":
        feat = df_local[[target_col]].copy()
    else:
        available = [c for c in base_cols if c in df_local.columns]
        feat = df_local[available].copy()
        if "Close" in df_local.columns:
            feat["close_return_1"] = df_local["Close"].pct_change().fillna(0.0)
        if "High" in df_local.columns and "Low" in df_local.columns:
            denom = df_local["Low"].replace(0, np.nan).bfill().ffill()
            feat["hl_spread_ratio"] = ((df_local["High"] - df_local["Low"]) / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    target = pd.to_numeric(df_local[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return feat, target


def to_sequences(x_2d: np.ndarray, y_1d: np.ndarray, lookback: int = 20) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert 2D features into sliding-window 3D sequences."""
    x_seq, y_seq = [], []
    for i in range(lookback, len(y_1d)):
        x_seq.append(x_2d[i - lookback : i])
        y_seq.append(y_1d[i])
    if not x_seq:
        return None, None
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def prepare_advanced_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    mode: str = "univariate",
    lookback: int = 20,
) -> Optional[Dict[str, Any]]:
    """Scale and sequence data for advanced forecasting models."""
    x_train_df, y_train = build_feature_frame(train_df, target_col, mode=mode)
    x_test_df, y_test = build_feature_frame(test_df, target_col, mode=mode)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train = x_scaler.fit_transform(x_train_df.to_numpy(dtype=float))
    x_test = x_scaler.transform(x_test_df.to_numpy(dtype=float))
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy(dtype=float).reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.to_numpy(dtype=float).reshape(-1, 1)).flatten()

    x_train_seq, y_train_seq = to_sequences(x_train, y_train_scaled, lookback=lookback)
    x_test_seq, y_test_seq = to_sequences(x_test, y_test_scaled, lookback=lookback)

    if x_train_seq is None or x_test_seq is None:
        return None

    return {
        "X_train_seq": x_train_seq,
        "y_train_seq": y_train_seq,
        "X_test_seq": x_test_seq,
        "y_test_seq": y_test_seq,
        "X_train_flat": x_train_seq.reshape(x_train_seq.shape[0], -1),
        "X_test_flat": x_test_seq.reshape(x_test_seq.shape[0], -1),
        "y_scaler": y_scaler,
        "x_scaler": x_scaler,
        "lookback": lookback,
        "n_features": x_train_seq.shape[-1],
    }


def _log(msg: str) -> None:
    """Print a timestamped progress message immediately to stdout."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_comprehensive_experiments(
    data_by_stock: Dict[str, Dict[str, pd.DataFrame]],
    forecast_columns: Sequence[str] = ("Close", "High", "Low", "Volatility"),
    decomposition_types: Sequence[str] = ("original", "decomposition+model", "hybrid"),
    model_types: Sequence[str] = ("ARIMA", "ExpSmoothing", "LSTM", "XGBoost", "RandomForest"),
) -> pd.DataFrame:
    """Run the notebook-style exhaustive decomposition/model experiment loop."""
    results_all: List[Dict[str, Any]] = []

    total_stocks = len(data_by_stock)
    total_combos = total_stocks * len(forecast_columns) * len(decomposition_types) * len(model_types)
    _log(f"[comprehensive] Starting: {total_stocks} stock(s), "
         f"{len(forecast_columns)} column(s), {len(decomposition_types)} decompositions, "
         f"{len(model_types)} models  →  up to {total_combos} combinations")

    combo_idx = 0

    for stock, split in data_by_stock.items():
        train_data = split["train"]
        test_data = split["test"]
        _log(f"[comprehensive] ── Stock: {stock}  (train={len(train_data)}, test={len(test_data)} rows)")

        for col in forecast_columns:
            if col not in train_data.columns or col not in test_data.columns:
                _log(f"[comprehensive]    ↳ Column '{col}' not found — skipping")
                continue

            train_series = pd.to_numeric(train_data[col], errors="coerce").dropna()
            test_series = pd.to_numeric(test_data[col], errors="coerce").dropna()
            if len(train_series) < 20 or len(test_series) < 5:
                _log(f"[comprehensive]    ↳ Column '{col}' too short (train={len(train_series)}, test={len(test_series)}) — skipping")
                continue

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_series.to_numpy().reshape(-1, 1)).flatten()
            test_scaled = scaler.transform(test_series.to_numpy().reshape(-1, 1)).flatten()
            ml_data = prepare_ml_data(pd.Series(train_scaled), pd.Series(test_scaled), n_lags=20)

            x_tr = ml_data.get("X_train") if ml_data else None
            y_tr = ml_data.get("y_train") if ml_data else None
            x_te = ml_data.get("X_test") if ml_data else None

            valid_ml = bool(
                x_tr is not None
                and y_tr is not None
                and x_te is not None
                and len(x_tr) > 0
                and len(x_te) > 0
                and len(x_tr) == len(y_tr)
            )

            for decomp_type in decomposition_types:
                for model_type in model_types:
                    combo_idx += 1
                    _log(f"[comprehensive] combo {combo_idx}/{total_combos} | "
                         f"{stock}/{col} | decomp={decomp_type} model={model_type}")
                    pred = None

                    if decomp_type == "original":
                        try:
                            if model_type == "ARIMA":
                                pred = train_arima_model(train_scaled, test_scaled)
                            elif model_type == "ExpSmoothing":
                                pred = train_exponential_smoothing(train_scaled, test_scaled)
                            elif model_type == "LSTM" and valid_ml:
                                pred = train_lstm_model(x_tr, y_tr, x_te)
                            elif model_type == "XGBoost" and valid_ml:
                                pred = train_xgboost_model(x_tr, y_tr, x_te)
                            elif model_type == "RandomForest" and valid_ml:
                                pred = train_random_forest_model(x_tr, y_tr, x_te)
                            else:
                                _log(f"[comprehensive]    ↳ Skipped (ML data invalid or model mismatch)")
                        except Exception as _exc:
                            _log(f"[comprehensive]    ✗ ERROR ({decomp_type}/{model_type}): {_exc}")
                            traceback.print_exc(file=sys.stdout)
                            pred = None
                    elif decomp_type == "hybrid" and valid_ml:
                        try:
                            pred = train_hybrid_model(train_scaled, test_scaled, x_tr, y_tr, x_te)
                        except Exception as _exc:
                            _log(f"[comprehensive]    ✗ ERROR (hybrid/{model_type}): {_exc}")
                            traceback.print_exc(file=sys.stdout)
                            pred = None
                    elif decomp_type == "decomposition+model":
                        try:
                            period = min(60, max(2, len(train_scaled) // 4))
                            try:
                                train_decomp = seasonal_decompose(
                                    pd.Series(train_scaled),
                                    model="additive",
                                    period=period,
                                    extrapolate_trend="freq",
                                )
                            except TypeError:
                                train_decomp = seasonal_decompose(
                                    pd.Series(train_scaled),
                                    model="additive",
                                    period=period,
                                )

                            full_series = np.concatenate([train_scaled, test_scaled])
                            try:
                                full_decomp = seasonal_decompose(
                                    pd.Series(full_series),
                                    model="additive",
                                    period=period,
                                    extrapolate_trend="freq",
                                )
                            except TypeError:
                                full_decomp = seasonal_decompose(
                                    pd.Series(full_series),
                                    model="additive",
                                    period=period,
                                )

                            trend_train = pd.Series(train_decomp.trend).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
                            seasonal_train = pd.Series(train_decomp.seasonal).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
                            residual_train = pd.Series(train_decomp.resid).ffill().bfill().fillna(0.0).to_numpy(dtype=float)

                            trend_test = pd.Series(full_decomp.trend).iloc[len(train_scaled) :].ffill().bfill().fillna(0.0).to_numpy(dtype=float)
                            seasonal_test = pd.Series(full_decomp.seasonal).iloc[len(train_scaled) :].ffill().bfill().fillna(0.0).to_numpy(dtype=float)
                            residual_test = pd.Series(full_decomp.resid).iloc[len(train_scaled) :].ffill().bfill().fillna(0.0).to_numpy(dtype=float)

                            trend_pred = None
                            seasonal_pred = None
                            residual_pred = None

                            if model_type == "ARIMA":
                                trend_pred = train_arima_model(trend_train, trend_test)
                                seasonal_pred = train_arima_model(seasonal_train, seasonal_test)
                                residual_pred = train_arima_model(residual_train, residual_test)
                            elif model_type == "ExpSmoothing":
                                trend_pred = train_exponential_smoothing(trend_train, trend_test)
                                seasonal_pred = train_exponential_smoothing(seasonal_train, seasonal_test)
                                residual_pred = train_exponential_smoothing(residual_train, residual_test)
                            elif model_type in {"LSTM", "XGBoost", "RandomForest"}:
                                trend_ml = prepare_ml_data(pd.Series(trend_train), pd.Series(trend_test), n_lags=15)
                                seasonal_ml = prepare_ml_data(pd.Series(seasonal_train), pd.Series(seasonal_test), n_lags=15)
                                residual_ml = prepare_ml_data(pd.Series(residual_train), pd.Series(residual_test), n_lags=15)

                                if trend_ml and seasonal_ml and residual_ml:
                                    if model_type == "LSTM":
                                        trend_pred = train_lstm_model(
                                            trend_ml["X_train"], trend_ml["y_train"], trend_ml["X_test"], epochs=100, verbose=0
                                        )
                                        seasonal_pred = train_lstm_model(
                                            seasonal_ml["X_train"], seasonal_ml["y_train"], seasonal_ml["X_test"], epochs=100, verbose=0
                                        )
                                        residual_pred = train_lstm_model(
                                            residual_ml["X_train"], residual_ml["y_train"], residual_ml["X_test"], epochs=100, verbose=0
                                        )
                                    elif model_type == "XGBoost":
                                        trend_pred = train_xgboost_model(
                                            trend_ml["X_train"], trend_ml["y_train"], trend_ml["X_test"]
                                        )
                                        seasonal_pred = train_xgboost_model(
                                            seasonal_ml["X_train"], seasonal_ml["y_train"], seasonal_ml["X_test"]
                                        )
                                        residual_pred = train_xgboost_model(
                                            residual_ml["X_train"], residual_ml["y_train"], residual_ml["X_test"]
                                        )
                                    elif model_type == "RandomForest":
                                        trend_pred = train_random_forest_model(
                                            trend_ml["X_train"], trend_ml["y_train"], trend_ml["X_test"]
                                        )
                                        seasonal_pred = train_random_forest_model(
                                            seasonal_ml["X_train"], seasonal_ml["y_train"], seasonal_ml["X_test"]
                                        )
                                        residual_pred = train_random_forest_model(
                                            residual_ml["X_train"], residual_ml["y_train"], residual_ml["X_test"]
                                        )
                                else:
                                    _log(f"[comprehensive]    ↳ Decomp ML data prep failed for one or more components")

                            if trend_pred is not None and seasonal_pred is not None and residual_pred is not None:
                                min_len = min(len(trend_pred), len(seasonal_pred), len(residual_pred))
                                pred = (
                                    0.5 * np.asarray(trend_pred[:min_len], dtype=float)
                                    + 0.25 * np.asarray(seasonal_pred[:min_len], dtype=float)
                                    + 0.25 * np.asarray(residual_pred[:min_len], dtype=float)
                                )
                            else:
                                _log(f"[comprehensive]    ↳ One or more component predictions returned None")
                        except Exception as _exc:
                            _log(f"[comprehensive]    ✗ ERROR (decomp+model/{model_type}): {_exc}")
                            traceback.print_exc(file=sys.stdout)
                            pred = None

                    if pred is None:
                        _log(f"[comprehensive]    ↳ No prediction produced — skipping")
                        continue

                    pred = np.asarray(pred, dtype=float)
                    if len(pred) > len(test_scaled):
                        pred = pred[: len(test_scaled)]
                    elif len(pred) < len(test_scaled):
                        pred = np.pad(pred, (0, len(test_scaled) - len(pred)), mode="edge")

                    pred_original = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                    test_original = test_series.to_numpy(dtype=float)[: len(pred_original)]
                    if len(pred_original) == 0 or len(test_original) == 0:
                        _log(f"[comprehensive]    ↳ Empty prediction — skipping")
                        continue

                    metrics = calculate_metrics(test_original, pred_original)
                    _log(f"[comprehensive]    ✓ RMSE={metrics.get('RMSE', float('nan')):.4f}  "
                         f"MAE={metrics.get('MAE', float('nan')):.4f}  "
                         f"MAPE={metrics.get('MAPE', float('nan')):.4f}")
                    results_all.append(
                        {
                            "Stock": stock,
                            "Column": col,
                            "Decomposition": decomp_type,
                            "Model": model_type,
                            **metrics,
                        }
                    )

    _log(f"[comprehensive] Done. Total results: {len(results_all)}")
    return pd.DataFrame(results_all)


def summarize_full_metrics_with_quantiles(
    y_train: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    quantile_forecasts: Optional[Dict[float, Sequence[float]]] = None,
    quantile_levels: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Expose the framework notebook metric suite from one place."""
    return compute_full_metrics(
        y_train=y_train,
        y_true=y_true,
        y_pred=y_pred,
        quantile_forecasts=quantile_forecasts,
        quantile_levels=quantile_levels,
    )


def check_time_series_backend_availability(
    configured_backends: Sequence[str],
    backend_registry: Any,
    import_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Return backend registration and availability diagnostics."""
    rows: List[Dict[str, Any]] = []
    import_map = import_map or {
        "autogluon_timeseries": "autogluon.timeseries",
        "statsforecast": "statsforecast",
        "flaml": "flaml",
        "chronos": "transformers",
    }

    for backend in configured_backends:
        registered = bool(backend_registry.has(backend))
        backend_available = bool(backend_registry.get(backend).is_available()) if registered else False

        direct_import = False
        import_name = import_map.get(backend)
        if import_name:
            try:
                __import__(import_name)
                direct_import = True
            except Exception:
                direct_import = False

        rows.append(
            {
                "backend": backend,
                "registered": registered,
                "backend_available": backend_available,
                "direct_import": direct_import,
            }
        )

    return pd.DataFrame(rows)


def run_extra_backend_benchmark(
    dataframe: pd.DataFrame,
    stock_col: str = "Stock",
    date_col: str = "Date",
    target_col: str = "Close",
    horizon: int = 30,
) -> pd.DataFrame:
    """Run StatsForecast and FLAML notebook-style benchmark on first stock."""
    stocks = sorted(dataframe[stock_col].dropna().unique().tolist())
    if not stocks:
        return pd.DataFrame()

    stock = stocks[0]
    df = dataframe[dataframe[stock_col] == stock][[date_col, target_col]].copy()
    df["ds"] = pd.to_datetime(df[date_col], errors="coerce")
    df["y"] = pd.to_numeric(df[target_col], errors="coerce")
    df["unique_id"] = stock
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    if len(df) <= horizon:
        return pd.DataFrame()

    regular = regularize_series_frequency(df[["unique_id", "ds", "y"]], frequency="B")
    train = regular.iloc[:-horizon].reset_index(drop=True)
    test = regular.iloc[-horizon:].reset_index(drop=True)

    rows: List[Dict[str, Any]] = []

    # StatsForecast
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive

        sf_models = [
            ("AutoARIMA", AutoARIMA(season_length=5)),
            ("AutoETS", AutoETS(season_length=5)),
            ("AutoTheta", AutoTheta(season_length=5)),
            ("SeasonalNaive", SeasonalNaive(season_length=5)),
        ]
        sf = StatsForecast(models=[m for _, m in sf_models], freq="B", n_jobs=1, fallback_model=Naive())
        sf_fcst = sf.forecast(df=train, h=horizon)

        for model_name, model_obj in sf_models:
            col = type(model_obj).__name__
            if col not in sf_fcst.columns:
                matches = [c for c in sf_fcst.columns if col.lower() in str(c).lower()]
                if not matches:
                    continue
                col = matches[0]

            pred = np.asarray(sf_fcst[col], dtype=float)[:horizon]
            metrics = compute_full_metrics(train["y"], test["y"], pred)
            rows.append({"stock": stock, "target": target_col, "model": f"SF_{model_name}", **metrics})
    except Exception:
        pass

    # FLAML
    try:
        from flaml import AutoML

        automl = AutoML()
        train_x = train[["ds"]].copy()
        train_y = train["y"].copy()
        test_x = test[["ds"]].copy()

        automl.fit(
            X_train=train_x,
            y_train=train_y,
            task="ts_forecast_regression",
            period=5,
            time_budget=60,
            metric="mape",
            verbose=0,
        )
        pred = np.asarray(automl.predict(test_x), dtype=float)[:horizon]
        metrics = compute_full_metrics(train["y"], test["y"], pred)
        rows.append({"stock": stock, "target": target_col, "model": f"FLAML_{automl.best_estimator}", **metrics})
    except Exception:
        pass

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Deep-Learning Models (BiLSTM, Transformer, Attention, Stacked Hybrid)
# ─────────────────────────────────────────────────────────────────────────────

def train_bilstm_advanced(
    x_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    x_test_seq: np.ndarray,
    epochs: int = 15,
    verbose: int = 0,
) -> Optional[np.ndarray]:
    """Train a Bidirectional LSTM on sequence data. Requires TensorFlow."""
    try:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import layers as klayers
    except Exception:
        return None
    try:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(x_train_seq.shape[1], x_train_seq.shape[2])),
            klayers.Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
        model.fit(
            x_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=min(32, len(x_train_seq)),
            verbose=verbose,
            validation_split=0.1,
        )
        return np.asarray(model.predict(x_test_seq, verbose=0).flatten(), dtype=float)
    except Exception:
        return None


def train_transformer_advanced(
    x_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    x_test_seq: np.ndarray,
    epochs: int = 15,
    d_model: int = 64,
    n_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.1,
    verbose: int = 0,
) -> Optional[np.ndarray]:
    """Train a Transformer encoder for time-series regression. Requires TensorFlow."""
    try:
        from tensorflow.keras import layers as klayers, Model
        from tensorflow.keras.optimizers import Adam
    except Exception:
        return None
    try:
        inp = klayers.Input(shape=(x_train_seq.shape[1], x_train_seq.shape[2]))
        x = klayers.Dense(d_model)(inp)
        kd = max(8, d_model // n_heads)
        attn = klayers.MultiHeadAttention(num_heads=n_heads, key_dim=kd)(x, x)
        x = klayers.LayerNormalization(epsilon=1e-6)(x + klayers.Dropout(dropout)(attn))
        ff = klayers.Dense(ff_dim, activation="relu")(x)
        ff = klayers.Dense(d_model)(ff)
        x = klayers.LayerNormalization(epsilon=1e-6)(x + klayers.Dropout(dropout)(ff))
        x = klayers.GlobalAveragePooling1D()(x)
        x = klayers.Dense(64, activation="relu")(x)
        out = klayers.Dense(1)(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
        model.fit(
            x_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=min(32, len(x_train_seq)),
            verbose=verbose,
            validation_split=0.1,
        )
        return np.asarray(model.predict(x_test_seq, verbose=0).flatten(), dtype=float)
    except Exception:
        return None


def train_attention_forecaster(
    x_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    x_test_seq: np.ndarray,
    epochs: int = 15,
    d_model: int = 64,
    n_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.1,
    verbose: int = 0,
) -> Optional[np.ndarray]:
    """LLM-attention-inspired forecaster with stacked self-attention blocks. Requires TensorFlow."""
    try:
        from tensorflow.keras import layers as klayers, Model
        from tensorflow.keras.optimizers import Adam
    except Exception:
        return None
    try:
        kd = max(8, d_model // n_heads)
        inp = klayers.Input(shape=(x_train_seq.shape[1], x_train_seq.shape[2]))
        x = klayers.Dense(d_model)(inp)
        for _ in range(2):
            attn = klayers.MultiHeadAttention(num_heads=n_heads, key_dim=kd)(x, x)
            x = klayers.LayerNormalization(epsilon=1e-6)(x + klayers.Dropout(dropout)(attn))
            ff = klayers.Dense(ff_dim, activation="gelu")(x)
            ff = klayers.Dense(d_model)(ff)
            x = klayers.LayerNormalization(epsilon=1e-6)(x + klayers.Dropout(dropout)(ff))
        x = klayers.GlobalAveragePooling1D()(x)
        x = klayers.Dense(64, activation="relu")(x)
        x = klayers.Dropout(dropout)(x)
        out = klayers.Dense(1)(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(learning_rate=8e-4), loss="mse")
        model.fit(
            x_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=min(32, len(x_train_seq)),
            verbose=verbose,
            validation_split=0.1,
        )
        return np.asarray(model.predict(x_test_seq, verbose=0).flatten(), dtype=float)
    except Exception:
        return None


def train_xgboost_sequence(
    x_train_flat: np.ndarray,
    y_train_seq: np.ndarray,
    x_test_flat: np.ndarray,
) -> Optional[np.ndarray]:
    """Train XGBoost on flattened sequence windows."""
    try:
        import xgboost as xgb
    except Exception:
        return None
    try:
        model = xgb.XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
        )
        model.fit(np.asarray(x_train_flat, dtype=float), np.asarray(y_train_seq, dtype=float), verbose=False)
        return np.asarray(model.predict(np.asarray(x_test_flat, dtype=float)), dtype=float)
    except Exception:
        return None


def train_hybrid_stacked_model(
    x_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    x_test_seq: np.ndarray,
    x_train_flat: np.ndarray,
    x_test_flat: np.ndarray,
    epochs: int = 12,
    verbose: int = 0,
) -> Optional[np.ndarray]:
    """Stacked hybrid: ARIMA + BiLSTM + XGBoost with Ridge meta-learner."""
    n = len(y_train_seq)
    if n < 40:
        return None
    split = max(int(n * 0.8), 25)
    if split >= n - 5:
        split = n - 5

    x_tr_seq = x_train_seq[:split]
    x_val_seq = x_train_seq[split:]
    y_tr = y_train_seq[:split]
    y_val = y_train_seq[split:]
    x_tr_flat = x_train_flat[:split]
    x_val_flat = x_train_flat[split:]

    arima_val = train_arima_model(y_tr, y_val, order=(2, 1, 2))
    arima_test = train_arima_model(y_train_seq, np.zeros(len(x_test_seq)), order=(2, 1, 2))

    bilstm_val = train_bilstm_advanced(x_tr_seq, y_tr, x_val_seq, epochs=epochs, verbose=verbose)
    bilstm_test = train_bilstm_advanced(x_train_seq, y_train_seq, x_test_seq, epochs=epochs, verbose=verbose)

    xgb_val = train_xgboost_sequence(x_tr_flat, y_tr, x_val_flat)
    xgb_test = train_xgboost_sequence(x_train_flat, y_train_seq, x_test_flat)

    candidates = [
        (arima_val, arima_test),
        (bilstm_val, bilstm_test),
        (xgb_val, xgb_test),
    ]

    valid_val, valid_test = [], []
    for p_val, p_test in candidates:
        if (p_val is not None and p_test is not None
                and len(p_val) == len(y_val)
                and len(p_test) == len(x_test_seq)):
            valid_val.append(p_val)
            valid_test.append(p_test)

    if not valid_val:
        return None
    if len(valid_val) == 1:
        return valid_test[0]

    p_val_mat = np.column_stack(valid_val)
    p_test_mat = np.column_stack(valid_test)
    meta = Ridge(alpha=1.0)
    meta.fit(p_val_mat, y_val)
    return np.asarray(meta.predict(p_test_mat), dtype=float)


def run_advanced_experiments(
    data_by_stock: Dict[str, Dict[str, Any]],
    targets: Sequence[str] = ("Close", "Volatility"),
    setups: Sequence[str] = ("univariate", "multivariate"),
    model_names: Sequence[str] = (
        "BiLSTM",
        "Transformer",
        "AttentionForecaster",
        "XGBoostSeq",
        "Hybrid_ARIMA_BiLSTM_XGB",
    ),
    lookback: int = 20,
    epochs: int = 12,
) -> pd.DataFrame:
    """Run advanced univariate/multivariate deep learning experiments.

    Parameters
    ----------
    data_by_stock : dict mapping stock name → {"train": DataFrame, "test": DataFrame}.
    targets : forecast target columns.
    setups : "univariate" and/or "multivariate".
    model_names : advanced models to evaluate.
    lookback : number of look-back steps for sequence models.
    epochs : training epochs for neural models.

    Returns
    -------
    DataFrame with columns: Stock, Target, Setup, Model, MAE, RMSE, MAPE, DA.
    """
    results: List[Dict[str, Any]] = []

    total_stocks = len(data_by_stock)
    total_combos = total_stocks * len(targets) * len(setups) * len(model_names)
    _log(f"[advanced_dl] Starting: {total_stocks} stock(s), {len(targets)} target(s), "
         f"{len(setups)} setup(s), {len(model_names)} model(s)  →  up to {total_combos} runs")
    _log(f"[advanced_dl] Models: {list(model_names)}")

    combo_idx = 0

    for stock, split in data_by_stock.items():
        train_df = split["train"]
        test_df = split["test"]
        _log(f"[advanced_dl] ── Stock: {stock}  (train={len(train_df)}, test={len(test_df)} rows)")

        for target_col in targets:
            if target_col not in train_df.columns or target_col not in test_df.columns:
                _log(f"[advanced_dl]    ↳ Target '{target_col}' not found — skipping")
                continue

            for setup in setups:
                _log(f"[advanced_dl]    Preparing data: target={target_col}, setup={setup}, lookback={lookback}")
                try:
                    prep = prepare_advanced_data(
                        train_df=train_df,
                        test_df=test_df,
                        target_col=target_col,
                        mode=setup,
                        lookback=lookback,
                    )
                except Exception as _exc:
                    _log(f"[advanced_dl]    ✗ Data prep FAILED: {_exc}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                if prep is None:
                    _log(f"[advanced_dl]    ↳ Data prep returned None (insufficient data) — skipping")
                    continue

                x_tr_seq = prep["X_train_seq"]
                y_tr_seq = prep["y_train_seq"]
                x_te_seq = prep["X_test_seq"]
                y_te_seq = prep["y_test_seq"]
                x_tr_flat = prep["X_train_flat"]
                x_te_flat = prep["X_test_flat"]
                y_scaler = prep["y_scaler"]
                n_features = prep["n_features"]

                _log(f"[advanced_dl]    Data ready: "
                     f"x_train={x_tr_seq.shape}, y_train={y_tr_seq.shape}, "
                     f"x_test={x_te_seq.shape}, features={n_features}")

                ep = epochs + 2 if setup == "multivariate" else epochs

                for model_name in model_names:
                    combo_idx += 1
                    _log(f"[advanced_dl] combo {combo_idx}/{total_combos} | "
                         f"{stock}/{target_col}/{setup} | model={model_name}  (epochs={ep})")
                    pred_scaled: Optional[np.ndarray] = None
                    try:
                        if model_name == "BiLSTM":
                            pred_scaled = train_bilstm_advanced(x_tr_seq, y_tr_seq, x_te_seq, epochs=ep)
                        elif model_name == "Transformer":
                            pred_scaled = train_transformer_advanced(x_tr_seq, y_tr_seq, x_te_seq, epochs=ep)
                        elif model_name == "AttentionForecaster":
                            pred_scaled = train_attention_forecaster(x_tr_seq, y_tr_seq, x_te_seq, epochs=ep)
                        elif model_name == "XGBoostSeq":
                            pred_scaled = train_xgboost_sequence(x_tr_flat, y_tr_seq, x_te_flat)
                        elif model_name == "Hybrid_ARIMA_BiLSTM_XGB":
                            pred_scaled = train_hybrid_stacked_model(
                                x_tr_seq, y_tr_seq, x_te_seq,
                                x_tr_flat, x_te_flat,
                                epochs=max(8, ep - 2),
                            )

                        if pred_scaled is None:
                            _log(f"[advanced_dl]    ↳ Model returned None (import error or insufficient data)")
                            continue

                        if len(pred_scaled) != len(y_te_seq):
                            _log(f"[advanced_dl]    ↳ Shape mismatch: pred={len(pred_scaled)}, expected={len(y_te_seq)} — skipping")
                            continue

                        y_true = y_scaler.inverse_transform(y_te_seq.reshape(-1, 1)).flatten()
                        y_pred = y_scaler.inverse_transform(
                            np.asarray(pred_scaled, dtype=float).reshape(-1, 1)
                        ).flatten()

                        metrics = calculate_metrics(y_true, y_pred)
                        _log(f"[advanced_dl]    ✓ RMSE={metrics.get('RMSE', float('nan')):.4f}  "
                             f"MAE={metrics.get('MAE', float('nan')):.4f}  "
                             f"MAPE={metrics.get('MAPE', float('nan')):.4f}")
                        results.append({
                            "Stock": stock,
                            "Target": target_col,
                            "Setup": setup,
                            "Model": model_name,
                            **metrics,
                            "Lookback": lookback,
                            "Features": n_features,
                        })
                    except Exception as _exc:
                        _log(f"[advanced_dl]    ✗ ERROR ({model_name}): {_exc}")
                        traceback.print_exc(file=sys.stdout)
                        continue

    _log(f"[advanced_dl] Done. Total results: {len(results)}")
    return pd.DataFrame(results)
