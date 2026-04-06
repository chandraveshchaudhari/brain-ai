"""Hybrid forecasting model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd


@dataclass
class HybridModel:
    """Hybrid model descriptor used by the AutoML runner."""

    decomposition: str
    base_model: str

    def name(self) -> str:
        return f"hybrid_{self.decomposition.lower()}_{self.base_model.lower()}"

    def forecast(
        self,
        y: Sequence[float],
        X: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
    ) -> Dict[str, object]:
        """Return metadata for planning and dispatching hybrid execution."""
        return {
            "hybrid_name": self.name(),
            "decomposition": self.decomposition,
            "base_model": self.base_model,
            "supports_covariates": True,
            "supports_multivariate": True,
            "input_rows": len(y),
            "has_exogenous": X is not None or future_covariates is not None,
        }


HYBRID_TEMPLATES = [
    HybridModel(decomposition="STL", base_model="NHITS"),
    HybridModel(decomposition="STL", base_model="RandomForest"),
    HybridModel(decomposition="ARIMA", base_model="XGBoost"),
    HybridModel(decomposition="ENSEMBLE", base_model="Blend"),
    HybridModel(decomposition="RESIDUAL", base_model="NHITS"),
]
