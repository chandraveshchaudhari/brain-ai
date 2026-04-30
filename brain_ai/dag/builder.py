from typing import Optional
import json
import os

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional
    nx = None


class DAGBuilder:
    """Builds a simple DAG representation for a PipelineSpec.

    The DAG nodes follow the pipeline flow:
    data -> granularity -> encoder -> fusion -> model -> evaluation
    """

    def __init__(self, out_dir: str = "."):
        self.out_dir = out_dir

    def build(self, spec) -> Optional[object]:
        if nx is None:
            return None
        G = nx.DiGraph()
        G.add_node("data")
        G.add_node("granularity")
        G.add_node("encoder")
        G.add_node("fusion")
        G.add_node("model")
        G.add_node("evaluation")

        G.add_edges_from([
            ("data", "granularity"),
            ("granularity", "encoder"),
            ("encoder", "fusion"),
            ("fusion", "model"),
            ("model", "evaluation"),
        ])

        # attach spec metadata
        G.graph["spec"] = json.loads(spec.to_json()) if hasattr(spec, "to_json") else {}
        return G

    def build_and_save(self, spec, filename: str = "pipeline_dag.png") -> str:
        G = self.build(spec)
        if G is None:
            raise RuntimeError("networkx not available")
        from .visualizer import save_graph_png

        path = os.path.join(self.out_dir, filename)
        save_graph_png(G, path)
        return path
