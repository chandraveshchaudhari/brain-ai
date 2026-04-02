"""Pipeline execution utilities for tool-driven workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class PipelineStep:
    """One executable step in a pipeline."""

    name: str
    callable_obj: Callable[..., Any]
    kwargs: Dict[str, Any] = field(default_factory=dict)


class PipelineRunner:
    """Executes steps in order and collects outputs."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def run(self, steps: List[PipelineStep]) -> List[Any]:
        outputs: List[Any] = []
        for index, step in enumerate(steps):
            output = step.callable_obj(**step.kwargs)
            outputs.append(output)
            self.history.append(
                {
                    "index": index,
                    "step": step.name,
                    "kwargs": dict(step.kwargs),
                    "output_type": type(output).__name__,
                }
            )
        return outputs
