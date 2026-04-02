"""Helpers for modality notebook scaffolding and hybrid-model planning."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SUPPORTED_MODALITIES: Sequence[str] = (
    "tabular",
    "text",
    "time_series",
    "image",
    "audio",
    "multimodal",
)

DECOMPOSITION_ALGORITHMS: Sequence[str] = (
    "stl",
    "wavelet",
    "emd",
    "ceemdan",
)

HYBRID_MODELING_ROADMAP: Sequence[Dict[str, Any]] = (
    {
        "order": 1,
        "name": "weighted_ensemble_stack",
        "summary": "Combine statistical + ML + deep learning models in weighted ensembles.",
        "focus": "ensemble_design",
    },
    {
        "order": 2,
        "name": "decomposition_hybrids",
        "summary": "Add decomposition-based hybrids: STL, wavelet decomposition, EMD/CEEMDAN where feasible.",
        "focus": "signal_decomposition",
        "algorithms": list(DECOMPOSITION_ALGORITHMS),
    },
    {
        "order": 3,
        "name": "residual_learning",
        "summary": "Add residual learning pipelines where deep models learn residuals from decomposition components.",
        "focus": "residual_modeling",
    },
    {
        "order": 4,
        "name": "backtesting_and_metrics",
        "summary": "Track per-model and ensemble metrics with backtesting.",
        "focus": "evaluation",
    },
)


def get_supported_modalities() -> List[str]:
    """Return the supported modality names for notebook scaffolding."""
    return list(SUPPORTED_MODALITIES)


def get_decomposition_algorithms() -> List[str]:
    """Return supported decomposition families for hybrid time-series planning."""
    return list(DECOMPOSITION_ALGORITHMS)


def get_hybrid_modeling_roadmap() -> List[Dict[str, Any]]:
    """Return a copy of the hybrid time-series implementation roadmap."""
    return deepcopy(list(HYBRID_MODELING_ROADMAP))


def get_modality_notebook_plan() -> Dict[str, Any]:
    """Return the modality notebook generation plan."""
    return {
        "modalities": get_supported_modalities(),
        "template_naming": "<modality>_template.ipynb",
        "format": "json_notebook",
    }


def build_modality_notebook_template(modality: str) -> Dict[str, Any]:
    """Build a JSON notebook template for a given modality."""
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported modality: {modality}")

    modality_title = modality.replace("_", " ").title()
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    f"# {modality_title} Modality Experiment",
                    "",
                    "Objective: add dataset loading, preprocessing, modeling, and evaluation for this modality.",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "import pandas as pd",
                    "import numpy as np",
                    f"print('Running {modality} template notebook')",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def create_modality_notebooks(
    output_dir: str | Path,
    modalities: Iterable[str] | None = None,
    overwrite: bool = False,
) -> List[Path]:
    """Create starter notebooks for the selected modalities."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    selected = list(modalities) if modalities is not None else get_supported_modalities()
    created_paths: List[Path] = []

    for modality in selected:
        notebook_path = target_dir / f"{modality}_template.ipynb"
        if notebook_path.exists() and not overwrite:
            created_paths.append(notebook_path)
            continue

        notebook_content = build_modality_notebook_template(modality)
        notebook_path.write_text(json.dumps(notebook_content, indent=2), encoding="utf-8")
        created_paths.append(notebook_path)

    return created_paths