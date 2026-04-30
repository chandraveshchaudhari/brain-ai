from abc import ABC, abstractmethod
from typing import Any


class BaseFusion(ABC):
    """Base interface for fusion strategies."""

    @abstractmethod
    def fuse(self, features: Any) -> Any:
        """Combine modality features into a fused representation.

        Expected to return a dict-like object with `X` and optionally `y`.
        """
        raise NotImplementedError()
