from abc import ABC, abstractmethod
from typing import Any


class BaseModelAdapter(ABC):
    """Adapter interface for different model backends.

    Adapters must be thin wrappers: no business logic, only translation
    between framework APIs and the system's expectations.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any = None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: Any) -> Any:
        raise NotImplementedError()
