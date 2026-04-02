"""Core architecture primitives for Brain-AI.

This package contains reusable contracts and runtime utilities used by
modalities, tools, and the LLM-driven pipeline layer.
"""

from brain_automl.core.pipeline import PipelineRunner
from brain_automl.core.protocols import BaseLibraryBackend, BaseModalityExecutor, BaseFusionStrategy, BaseTool
from brain_automl.core.registry import Registry, BACKEND_REGISTRY, TOOL_REGISTRY
from brain_automl.core.result import ModalityResult, FusionResult

__all__ = [
    "PipelineRunner",
    "Registry",
    "BaseLibraryBackend",
    "BaseModalityExecutor",
    "BaseFusionStrategy",
    "BaseTool",
    "BACKEND_REGISTRY",
    "TOOL_REGISTRY",
    "ModalityResult",
    "FusionResult",
]
