"""Abstract contracts for extensible Brain-AI architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


class BaseLibraryBackend(ABC):
    """Contract for one concrete model backend (for example AutoGluon or FLAML)."""

    name: str = "backend"
    modality: str = "generic"
    task_types: Iterable[str] = ()

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when runtime dependencies are installed and usable."""

    @abstractmethod
    def fit(self, x_train: Any, y_train: Any, **kwargs: Any) -> Any:
        """Train and return a model object or backend-specific handle."""

    @abstractmethod
    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        """Return predictions for input data."""


class BaseModalityExecutor(ABC):
    """Contract for a modality orchestrator (tabular/text/time_series/image/audio)."""

    modality: str = "generic"

    @abstractmethod
    def run(self, data: Any, task: str, **kwargs: Any) -> List[Any]:
        """Run selected backends and return normalized modality results."""


class BaseFusionStrategy(ABC):
    """Contract for decision, feature, or information fusion implementations."""

    name: str = "fusion"

    @abstractmethod
    def fuse(self, results: List[Any], **kwargs: Any) -> Any:
        """Fuse multiple modality results into one combined result."""


class BaseTool(ABC):
    """Contract for LLM-callable tools with explicit JSON-schema IO."""

    name: str = "tool"
    description: str = ""

    @classmethod
    @abstractmethod
    def input_schema(cls) -> Dict[str, Any]:
        """JSON schema used by planner/tool-calling layers."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute tool and return structured output."""
