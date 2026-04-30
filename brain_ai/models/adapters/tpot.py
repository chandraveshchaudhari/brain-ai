from ..base import BaseModelAdapter
from typing import Any


class TPOTAdapter(BaseModelAdapter):
    """Scaffold adapter for TPOT pipelines."""

    def __init__(self, tpot_estimator=None):
        self.tpot = tpot_estimator

    def fit(self, X: Any, y: Any = None):
        if self.tpot is None:
            raise RuntimeError("TPOT estimator not provided")
        self.tpot.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self.tpot.predict(X)
