from typing import Dict, Any
from ...decision.spec import PipelineSpec


def generate_pipeline(task_description: str) -> Dict[str, Any]:
    """Generate a PipelineSpec from a task description (deterministic scaffold).

    Skills must accept structured input and return structured output.
    This function is a safe, testable scaffold; LLM layers should call it
    instead of executing arbitrary code.
    """
    # VERY simple heuristic mapping (real LLM integration should be separate)
    spec = PipelineSpec(
        modalities=["tabular"],
        granularity_strategy="resample",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={},
    )
    return {
        "status": "ok",
        "task_description": task_description,
        "pipeline_spec": spec.to_dict(),
    }
