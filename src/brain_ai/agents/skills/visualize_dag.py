from typing import Any, Dict

from ...decision.spec import PipelineSpec


def visualize_dag(spec: PipelineSpec, dag_builder, out_path: str = "pipeline_dag.png") -> Dict[str, Any]:
    """Create DAG image using the provided dag_builder and return path."""
    path = dag_builder.build_and_save(spec, filename=out_path)
    return {
        "status": "ok",
        "pipeline_spec": spec.to_dict(),
        "dag_path": path,
    }
