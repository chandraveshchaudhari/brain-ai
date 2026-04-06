"""Multimodal forecasting interface for future-ready expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MultimodalForecastEngine:
    """Container for multimodal forecasting inputs and training plans."""

    def fit(
        self,
        time_series: Any,
        tabular: Any = None,
        text: Any = None,
        image: Any = None,
    ) -> Dict[str, Any]:
        """Build a multimodal training plan based on provided modalities."""
        active_modalities = ["time_series"]
        if tabular is not None:
            active_modalities.append("tabular")
        if text is not None:
            active_modalities.append("text")
        if image is not None:
            active_modalities.append("image")

        return {
            "active_modalities": active_modalities,
            "fusion_ready": len(active_modalities) >= 2,
            "status": "planned",
        }
