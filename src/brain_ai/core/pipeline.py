"""Backward-compatible re-export for PipelineSpec.

Prefer importing PipelineSpec from brain_ai.decision.spec.
"""

from ..decision.spec import PipelineSpec

__all__ = ["PipelineSpec"]
