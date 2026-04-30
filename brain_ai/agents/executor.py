from typing import Any, Dict


class Executor:
    """Executes PipelineSpec using provided components.

    The Executor accepts DI components such as granularity, fusion, model,
    evaluator and dag_builder. It returns metrics and artifacts.
    """

    def __init__(self, brain):
        self.brain = brain

    def execute(self, spec: Dict[str, Any], data: Any) -> Dict[str, Any]:
        # expect spec to be a PipelineSpec-like object
        return self.brain.run_pipeline(spec, data)
