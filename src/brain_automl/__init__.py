"""Brain-AI package root.

This module intentionally keeps legacy imports stable while exposing new core
architecture utilities for ongoing migration.
"""

from brain_automl.config import DEFAULT_CONFIG, get_default_config
from brain_automl.core import (
	BACKEND_REGISTRY,
	TOOL_REGISTRY,
	FusionResult,
	ModalityResult,
	PipelineRunner,
)
from brain_automl.legacy_bridge import Brain, LegacyBrainBridge

__all__ = [
	"DEFAULT_CONFIG",
	"get_default_config",
	"BACKEND_REGISTRY",
	"TOOL_REGISTRY",
	"ModalityResult",
	"FusionResult",
	"PipelineRunner",
	"Brain",
	"LegacyBrainBridge",
]

