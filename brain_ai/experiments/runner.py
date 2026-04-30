from typing import List, Dict, Any


class Runner:
    """Run multiple PipelineSpecs and collect results."""

    def __init__(self, executor):
        self.executor = executor

    def run_batch(self, specs: List[Any], data: Any) -> List[Dict[str, Any]]:
        results = []
        for spec in specs:
            res = self.executor.execute(spec, data)
            results.append(res)
        return results
