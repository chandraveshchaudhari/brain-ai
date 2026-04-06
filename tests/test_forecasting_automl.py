"""Tests for unified forecasting AutoML runner and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from brain_automl.automl import BrainAutoMLForecast
from brain_automl.metrics import compute_forecasting_metrics, directional_accuracy


def test_directional_accuracy_basic_case():
    y_true = [10, 11, 10, 12]
    y_pred = [9, 10, 9, 11]
    # True directions: +, -, + ; Pred directions: +, -, + => 1.0
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_compute_forecasting_metrics_has_required_keys():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.2, 3.8]

    metrics = compute_forecasting_metrics(y_true, y_pred)
    assert set(["mse", "rmse", "mae", "mape", "directional_accuracy"]).issubset(metrics.keys())


def test_brain_automl_sorts_by_mse_then_directional_accuracy():
    runner = BrainAutoMLForecast(metric="mse", secondary_metric="directional_accuracy")
    board = pd.DataFrame(
        [
            {"model": "a", "mse": 1.0, "directional_accuracy": 0.7, "status": "ok"},
            {"model": "b", "mse": 0.9, "directional_accuracy": 0.6, "status": "ok"},
            {"model": "c", "mse": 0.9, "directional_accuracy": 0.8, "status": "ok"},
        ]
    )
    sorted_board = runner._sort_leaderboard(board)
    assert sorted_board.iloc[0]["model"] == "c"


def test_brain_automl_multimodal_fit_contract():
    runner = BrainAutoMLForecast()
    ts_df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=5, freq="D"), "y": np.arange(5)})
    out = runner.fit(time_series=ts_df, tabular=pd.DataFrame({"f1": [1, 2, 3, 4, 5]}))

    assert "time_series" in out["active_modalities"]
    assert "tabular" in out["active_modalities"]
    assert out["fusion_ready"] is True


def test_brain_automl_run_multi_horizon_aggregates_results(monkeypatch):
    runner = BrainAutoMLForecast()

    def _fake_run(self, **kwargs):
        hz = kwargs["horizon"]
        lb = pd.DataFrame(
            [
                {
                    "model": f"m_{hz}",
                    "library": "framework",
                    "mse": float(hz),
                    "directional_accuracy": 0.5,
                    "status": "ok",
                }
            ]
        )
        return {
            "leaderboard": lb,
            "best_model": lb.iloc[0].to_dict(),
            "predictions": {},
            "output_dir": "tmp",
            "metric": "mse",
            "secondary_metric": "directional_accuracy",
        }

    monkeypatch.setattr(BrainAutoMLForecast, "run", _fake_run)

    out = runner.run_multi_horizon(
        data=pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=10, freq="D"), "y": np.arange(10)}),
        timestamp_column="ds",
        target_column="y",
        horizons=[7, 1, 30],
    )

    assert out["horizons"] == [7, 1, 30]
    assert set(out["runs_by_horizon"].keys()) == {1, 7, 30}
    assert not out["leaderboard"].empty
    assert sorted(out["leaderboard"]["horizon"].unique().tolist()) == [1, 7, 30]
    assert sorted(out["best_by_horizon"]["horizon"].tolist()) == [1, 7, 30]
