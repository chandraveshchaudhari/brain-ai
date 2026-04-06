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
from brain_automl.automl import BrainAutoMLForecast
from brain_automl.forecasting import (
	HYBRID_TEMPLATES,
	HierarchicalForecast,
	HybridModel,
	MultimodalForecastEngine,
	SUPPORTED_HIERARCHICAL_METHODS,
	list_deep_learning_models,
	list_foundation_models,
)
from brain_automl.legacy_bridge import Brain, LegacyBrainBridge
from brain_automl.utilities import (
	build_modality_notebook_template,
	create_modality_notebooks,
	get_decomposition_algorithms,
	get_hybrid_modeling_roadmap,
	get_modality_notebook_plan,
	get_supported_modalities,
)

__all__ = [
	"DEFAULT_CONFIG",
	"get_default_config",
	"BACKEND_REGISTRY",
	"TOOL_REGISTRY",
	"ModalityResult",
	"FusionResult",
	"PipelineRunner",
	"BrainAutoMLForecast",
	"Brain",
	"LegacyBrainBridge",
	"HybridModel",
	"HierarchicalForecast",
	"MultimodalForecastEngine",
	"HYBRID_TEMPLATES",
	"SUPPORTED_HIERARCHICAL_METHODS",
	"list_foundation_models",
	"list_deep_learning_models",
	"build_modality_notebook_template",
	"create_modality_notebooks",
	"get_decomposition_algorithms",
	"get_hybrid_modeling_roadmap",
	"get_modality_notebook_plan",
	"get_supported_modalities",
]

