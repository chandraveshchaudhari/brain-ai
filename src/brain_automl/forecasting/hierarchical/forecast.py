"""Hierarchical forecasting planning interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class HierarchicalForecast:
    """Descriptor for hierarchical forecasting reconciliation."""

    method: str = "bottom_up"

    def plan(self) -> Dict[str, str]:
        """Return execution metadata for hierarchical workflows."""
        return {
            "method": self.method,
            "library": "darts",
            "reconciliation_supported": "yes",
        }


SUPPORTED_HIERARCHICAL_METHODS = [
    "bottom_up",
    "top_down",
    "middle_out",
    "reconciliation",
]
