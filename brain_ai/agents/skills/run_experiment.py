from typing import Dict, Any
from ...decision.spec import PipelineSpec


def run_experiment(spec: PipelineSpec, components: Dict[str, Any], data: Any) -> Dict[str, Any]:
    """Run pipeline experiment by delegating to provided components.

    `components` should include `granularity`, `fusion`, `model_adapter`,
    `evaluator`, and optionally `dag_builder`.
    """
    brain = components.get("brain")
    if brain is None:
        # construct minimal brain if components provided
        from ...core.brain import Brain

        brain = Brain(
            granularity=components.get("granularity"),
            fusion=components.get("fusion"),
            model_adapter=components.get("model_adapter"),
            evaluator=components.get("evaluator"),
            dag_builder=components.get("dag_builder"),
        )

    result = brain.run_pipeline(spec, data)
    return {
        "status": "ok",
        "pipeline_spec": spec.to_dict(),
        "result": result,
    }
