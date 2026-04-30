from .datasets import generate_multimodal_regression_data

__all__ = ["generate_multimodal_regression_data"]
"""Utility helpers for Brain-AI (keep minimal)."""

from typing import Dict


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]
