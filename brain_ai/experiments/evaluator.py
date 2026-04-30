from typing import Any, Dict
import numpy as np


class Evaluator:
    """Lightweight evaluator returning example metrics.

    Research code should replace with domain-appropriate metrics.
    """

    def evaluate(self, preds: Any, y: Any = None) -> Dict[str, float]:
        preds_arr = np.asarray(preds)
        if y is None:
            return {"score": float(preds_arr.mean())}

        y_arr = np.asarray(y)
        mse = float(np.mean((preds_arr - y_arr) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds_arr - y_arr)))
        score = -rmse
        return {"score": score, "rmse": rmse, "mae": mae}
