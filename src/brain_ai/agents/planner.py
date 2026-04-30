from ..decision.spec import PipelineSpec


class Planner:
    """Simple planner that turns a task description into a PipelineSpec.

    This is intentionally lightweight; real LLM-driven planners should call
    into `agents.skills.generate_pipeline` instead of reimplementing logic.
    """

    def plan(self, task: str) -> PipelineSpec:
        # scaffold: convert text to a minimal structured specification
        return PipelineSpec(
            modalities=["tabular"],
            granularity_strategy="resample",
            fusion_strategy="early",
            model_backend="sklearn",
            hyperparameters={},
        )
