"""Forecasting pipeline wrapper around the unified AutoML engine."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import pandas as pd

from brain_automl.automl import BrainAutoMLForecast


def run_forecasting_pipeline(
    data: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    item_id_column: Optional[str] = None,
    horizon: int = 252,
    exogenous_columns: Optional[Sequence[str]] = None,
    multiple_targets: Optional[Sequence[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run end-to-end forecasting benchmark pipeline."""
    runner = BrainAutoMLForecast(config=config)
    return runner.run(
        data=data,
        timestamp_column=timestamp_column,
        target_column=target_column,
        item_id_column=item_id_column,
        horizon=horizon,
        exogenous_columns=exogenous_columns,
        multiple_targets=multiple_targets,
    )
