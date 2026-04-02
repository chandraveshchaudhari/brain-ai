"""Standard result objects shared across all modalities and fusion layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModalityResult:
    """Normalized output for a single modality + backend execution."""

    modality: str
    backend: str
    task: str
    predictions: Any
    probabilities: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Normalized output for fused results across multiple modalities."""

    strategy: str
    predictions: Any
    probabilities: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
