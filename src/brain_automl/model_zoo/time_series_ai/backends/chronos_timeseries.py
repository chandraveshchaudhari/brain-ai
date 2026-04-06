"""Chronos time-series backend adapter.

This backend mirrors the benchmark-style Chronos usage from examples while
adhering to the framework backend contract.
"""

from __future__ import annotations

import re
from typing import Any, List

import numpy as np
import pandas as pd

from brain_automl.core.protocols import BaseLibraryBackend
from brain_automl.core.registry import BACKEND_REGISTRY


@BACKEND_REGISTRY.register("chronos")
class ChronosTimeSeriesBackend(BaseLibraryBackend):
    name = "chronos"
    modality = "time_series"
    task_types = ("forecasting",)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _extract_numeric_values(text: str) -> List[float]:
        """Parse floating-point values from generated Chronos text output."""
        matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        return [float(match) for match in matches]

    def fit(self, x_train: Any, y_train: Any = None, **kwargs: Any) -> Any:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = str(kwargs.get("chronos_model") or "amazon/chronos-t5-small")
        prediction_length = int(kwargs.get("prediction_length") or kwargs.get("horizon") or 14)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        return {
            "backend": self.name,
            "model_name": model_name,
            "tokenizer": tokenizer,
            "model": model,
            "series": x_train["y"].astype(float).tolist(),
            "prediction_length": prediction_length,
            "item_id": x_train["unique_id"].iloc[-1] if len(x_train) else "series1",
        }

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        series = model["series"]
        if not series:
            raise ValueError("Chronos backend received empty training series")

        tokenizer = model["tokenizer"]
        seq2seq_model = model["model"]
        prediction_length = int(model.get("prediction_length", len(x_test)))

        input_text = ",".join(map(str, series))
        inputs = tokenizer(input_text, return_tensors="pt")

        generated = seq2seq_model.generate(
            **inputs,
            max_new_tokens=max(prediction_length * 2, 32),
        )
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        numeric_values = self._extract_numeric_values(decoded)

        if not numeric_values:
            numeric_values = [float(series[-1])] * prediction_length
        elif len(numeric_values) < prediction_length:
            numeric_values.extend([float(numeric_values[-1])] * (prediction_length - len(numeric_values)))
        elif len(numeric_values) > prediction_length:
            numeric_values = numeric_values[:prediction_length]

        pred = np.asarray(numeric_values, dtype=float)
        expected_len = len(x_test)
        if expected_len > 0:
            if len(pred) > expected_len:
                pred = pred[:expected_len]
            elif len(pred) < expected_len:
                pred = np.pad(pred, (0, expected_len - len(pred)), mode="edge")

        return pd.DataFrame(
            {
                "unique_id": x_test["unique_id"].values,
                "ds": pd.to_datetime(x_test["ds"].values),
                "prediction": pred,
            }
        )
