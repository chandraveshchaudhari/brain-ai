"""Utilities for summarizing forecasting experiment outcomes."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def summarize_forecasting_experiment(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact summary statistics from BrainAutoMLForecast output."""
    leaderboard = result.get("leaderboard")
    if leaderboard is None or not isinstance(leaderboard, pd.DataFrame) or leaderboard.empty:
        return {
            "has_results": False,
            "n_models": 0,
            "best_model": None,
        }

    best = leaderboard.iloc[0]
    return {
        "has_results": True,
        "n_models": int(len(leaderboard)),
        "best_model": str(best.get("model")),
        "best_library": str(best.get("library")),
        "best_mse": float(best.get("mse")),
        "best_directional_accuracy": float(best.get("directional_accuracy")),
    }
