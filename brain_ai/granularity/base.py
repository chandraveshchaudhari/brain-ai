from abc import ABC, abstractmethod
from typing import Any


class BaseGranularity(ABC):
    """Base interface for granularity strategies.

    Implementations must provide an `align` method which accepts raw
    multimodal data and returns an aligned structure (dict-like) with
    at minimum features `X` and optional labels `y`.
    """

    @abstractmethod
    def align(self, data: Any) -> Any:
        """Align input data to the desired granularity.

        Returns a dict-like object with keys `X` and optionally `y`.
        """
        raise NotImplementedError()
