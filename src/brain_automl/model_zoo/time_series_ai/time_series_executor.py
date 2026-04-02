"""Registry-driven time-series executor."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from brain_automl.config import get_default_config
from brain_automl.core.protocols import BaseModalityExecutor
from brain_automl.core.registry import BACKEND_REGISTRY
from brain_automl.core.result import ModalityResult
from brain_automl.model_zoo.time_series_ai import backends  # noqa: F401


class TimeSeriesAutoML(BaseModalityExecutor):
    """Runs configured time-series backends and normalizes outputs."""

    modality = "time_series"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or get_default_config()

    def _selected_backends(self) -> List[str]:
        backend_cfg = self.config["backends"]["by_modality"]["time_series"]
        return list(backend_cfg.get("default", []))

    def run(
        self,
        data: pd.DataFrame,
        task: str = "forecasting",
        backends: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[ModalityResult]:
        selected = list(backends) if backends is not None else self._selected_backends()
        outputs: List[ModalityResult] = []

        for backend_name in selected:
            if not BACKEND_REGISTRY.has(backend_name):
                if self.config["backends"]["skip_unavailable_backends"]:
                    continue
                raise KeyError(f"Time-series backend '{backend_name}' is not registered")

            backend_cls = BACKEND_REGISTRY.get(backend_name)
            if hasattr(backend_cls, "modality") and backend_cls.modality != self.modality:
                continue

            if not backend_cls.is_available():
                if self.config["backends"]["skip_unavailable_backends"]:
                    continue
                raise RuntimeError(f"Backend '{backend_name}' is not available in this environment")

            backend = backend_cls()
            model = backend.fit(data, None, task=task, **kwargs)
            prediction = backend.predict(model, data, task=task, **kwargs)
            outputs.append(
                ModalityResult(
                    modality=self.modality,
                    backend=backend_name,
                    task=task,
                    predictions=prediction,
                    metrics={},
                    metadata={"status": "ok"},
                )
            )

        if not outputs and self.config["backends"]["hard_fail_if_no_backend_available_for_modality"]:
            raise RuntimeError("No available time-series backend produced output")

        return outputs
