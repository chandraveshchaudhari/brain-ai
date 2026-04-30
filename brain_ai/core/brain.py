from typing import Any, Dict, Optional

from ..decision.spec import PipelineSpec

try:
    import mlflow
except Exception:  # pragma: no cover - optional
    mlflow = None


class Brain:
    """Orchestrator for pipelines built from PipelineSpec.

    The Brain class wires granularity, fusion and model adapter components
    using dependency injection. Implementations should be passed in at
    runtime (no hard imports here) to keep the orchestrator open for
    extension.
    """

    def __init__(
        self,
        granularity=None,
        fusion=None,
        model_adapter=None,
        evaluator=None,
        dag_builder=None,
        decision_engine=None,
    ):
        self.granularity = granularity
        self.fusion = fusion
        self.model_adapter = model_adapter
        self.evaluator = evaluator
        self.dag_builder = dag_builder
        self.decision_engine = decision_engine

    def _resolve_component(self, direct_component, registry: Optional[Dict[str, Any]], key: str):
        # Explicit user override always wins.
        if direct_component is not None:
            return direct_component
        if registry is None:
            return None
        return registry.get(key)

    def run_pipeline(self, spec: PipelineSpec, raw_data: Any) -> Dict[str, Any]:
        if isinstance(spec, dict):
            spec = PipelineSpec.from_dict(spec)

        # log spec
        if mlflow is not None:
            spec.log_mlflow()

        granularity = self.granularity
        fusion = self.fusion
        model_adapter = self.model_adapter

        if self.decision_engine is not None:
            chosen = self.decision_engine.resolve_components(spec)
            granularity = self._resolve_component(granularity, chosen, "granularity")
            fusion = self._resolve_component(fusion, chosen, "fusion")
            model_adapter = self._resolve_component(model_adapter, chosen, "model_adapter")

        # 1. Granularity alignment
        if granularity is None:
            raise RuntimeError("Granularity component is required")
        aligned = granularity.align(raw_data)

        # 2. Extract features / encoders assumed to be done upstream in aligned
        # 3. Fusion
        if fusion is None:
            raise RuntimeError("Fusion component is required")
        fused = fusion.fuse(aligned)

        # 4. Model fit and predict
        model = model_adapter
        if model is None:
            raise RuntimeError("Model adapter is required")
        model.fit(fused["X"], fused.get("y"))
        preds = model.predict(fused["X"])

        # 5. Evaluate
        metrics = {}
        if self.evaluator is not None:
            metrics = self.evaluator.evaluate(preds, fused.get("y"))

        # 6. DAG logging
        if self.dag_builder is not None and mlflow is not None:
            path = self.dag_builder.build_and_save(spec)
            mlflow.log_artifact(path, artifact_path="dag")

        # log metrics
        if mlflow is not None:
            mlflow.log_metrics(metrics)

        return {"predictions": preds, "metrics": metrics}
