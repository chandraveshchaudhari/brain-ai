from ..base import BaseModelAdapter
from typing import Any

try:
    from sklearn.base import clone
except Exception:  # pragma: no cover - optional
    clone = None


class SKLearnAdapter(BaseModelAdapter):
    """Thin adapter for sklearn estimator instances.

    Example:
        adapter = SKLearnAdapter(estimator=RandomForestRegressor())
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X: Any, y: Any = None):
        self.estimator.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self.estimator.predict(X)
