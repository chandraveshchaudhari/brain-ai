from ..base import BaseModelAdapter
from typing import Any


class AutoGluonAdapter(BaseModelAdapter):
    """Scaffold adapter for AutoGluon (thin wrapper).

    Real integration should isolate AutoGluon imports and be optional.
    """

    def __init__(self, predictor=None):
        self.predictor = predictor

    def fit(self, X: Any, y: Any = None):
        if self.predictor is None:
            raise RuntimeError("AutoGluon predictor not provided")
        self.predictor.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self.predictor.predict(X)
