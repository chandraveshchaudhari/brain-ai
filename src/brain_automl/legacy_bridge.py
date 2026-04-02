"""Legacy Bridge: backward-compatible shims for Brain and TabularAutoML.

This module ensures existing code continues working while internally using
the new registry-based architecture. No API changes for existing users.
"""

import warnings
from typing import Any, Dict, Optional, Type

import pandas as pd

from brain_automl.core import ModalityResult, BACKEND_REGISTRY, BaseLibraryBackend
from brain_automl.model_zoo.time_series_ai import TimeSeriesAutoML


class LegacyBrainBridge:
    """Backward-compatible wrapper for the legacy Brain API.
    
    Routes calls through the new registry-based executor while maintaining
    the old interface. All existing code continues to work unchanged.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the legacy bridge.
        
        Args:
            config: Optional configuration dict (ignored for now, will be passed
                   to new executor in Phase 2).
        """
        self.config = config or {}
        self._executors = {}
        warnings.warn(
            "LegacyBrainBridge is using the new registry-based architecture. "
            "Consider migrating to TimeSeriesAutoML() for best results.",
            DeprecationWarning,
            stacklevel=2
        )

    def run_time_series(
        self,
        data: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
        task: str = "forecasting",
        **kwargs
    ) -> Dict[str, Any]:
        """Run time-series forecasting using new executor.
        
        Args:
            data: Input DataFrame with time-series data.
            timestamp_col: Name of timestamp column.
            target_col: Name of target column.
            task: Task type ('forecasting', 'anomaly_detection', etc.).
            **kwargs: Additional backend-specific kwargs.
        
        Returns:
            Dict with results from all available backends. Each backend's
            output is in the standardized ModalityResult format.
        """
        if "executor" not in self._executors:
            self._executors["executor"] = TimeSeriesAutoML(self.config)

        executor = self._executors["executor"]

        # Ensure timestamp is in data
        if timestamp_col not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        try:
            results = executor.run(data, task=task, **kwargs)
        except Exception as e:
            warnings.warn(f"TimeSeriesAutoML execution failed: {e}", RuntimeWarning)
            return {"error": str(e), "results": []}

        # Convert new format to legacy-compatible dict
        return {
            "success": len(results) > 0,
            "results": [
                {
                    "backend": r.backend,
                    "predictions": r.predictions,
                    "probabilities": r.probabilities,
                    "metrics": r.metrics,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }

    def list_available_backends(self) -> Dict[str, bool]:
        """List available backends and their status.
        
        Returns:
            Dict mapping backend name to availability (True/False).
        """
        available = {}
        for name, backend_cls in BACKEND_REGISTRY.items().items():
            # backend_cls is a class reference that implements BaseLibraryBackend
            if not issubclass(backend_cls, BaseLibraryBackend):
                continue  # Skip non-backend entries
            available[name] = backend_cls.is_available()
        return available

    def get_executor(self, modality: str):
        """Get executor for a modality (future use).
        
        Args:
            modality: Modality name ('time_series', 'tabular', 'text', etc.).
        
        Returns:
            Executor instance (not yet implemented for all modalities).
        """
        if modality == "time_series":
            if "executor" not in self._executors:
                self._executors["executor"] = TimeSeriesAutoML(self.config)
            return self._executors["executor"]
        else:
            raise NotImplementedError(f"Executor for '{modality}' not yet implemented")


# Keep the legacy Brain class for drop-in compatibility
class Brain(LegacyBrainBridge):
    """Backward-compatible Brain class.
    
    This is the legacy Brain API, now implemented via LegacyBrainBridge.
    All existing code using Brain() continues to work unchanged.
    """

    pass
