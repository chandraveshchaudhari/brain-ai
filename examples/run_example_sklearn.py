"""Example: run a simple sklearn pipeline end-to-end using Brain-AI scaffolding.

This script demonstrates a minimal flow using the package's base
components: PipelineSpec -> Brain -> DAG logging -> model adapter.
"""
from pprint import pprint

import numpy as np

from src.brain_ai.decision.spec import PipelineSpec
from src.brain_ai.core.brain import Brain
from src.brain_ai.granularity.resample import ResampleGranularity
from src.brain_ai.fusion.early import EarlyFusion
from src.brain_ai.models.adapters.sklearn import SKLearnAdapter
from src.brain_ai.dag.builder import DAGBuilder

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class RMSEEvaluator:
    def evaluate(self, preds, y):
        preds = np.asarray(preds)
        y = np.asarray(y)
        mse = np.mean((preds - y) ** 2)
        return {"rmse": float(np.sqrt(mse))}


def main():
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    spec = PipelineSpec(
        modalities=["tabular"],
        granularity_strategy="resample",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={"n_estimators": 10},
    )

    gran = ResampleGranularity()
    fusion = EarlyFusion()
    model_adapter = SKLearnAdapter(estimator=RandomForestRegressor(n_estimators=10, random_state=0))
    evaluator = RMSEEvaluator()
    dag_builder = DAGBuilder(out_dir="examples")

    brain = Brain(granularity=gran, fusion=fusion, model_adapter=model_adapter, evaluator=evaluator, dag_builder=dag_builder)

    # Brain expects multimodal input as modality->array with optional labels.
    raw = {"modalities": {"tabular": X_train}, "y": y_train}

    print("Running pipeline (training on train split)...")
    result = brain.run_pipeline(spec, raw)
    pprint(result.get("metrics", {}))

    # Evaluate on holdout
    test_preds = model_adapter.predict(X_test)
    test_metrics = evaluator.evaluate(test_preds, y_test)
    print("Holdout metrics:")
    pprint(test_metrics)

    print("DAG image saved under examples/")


if __name__ == "__main__":
    main()
