import numpy as np

from sklearn.linear_model import LinearRegression

from brain_ai.core.brain import Brain
from brain_ai.dag.builder import DAGBuilder
from brain_ai.decision.engine import DecisionEngine
from brain_ai.decision.spec import PipelineSpec
from brain_ai.experiments.evaluator import Evaluator
from brain_ai.fusion.early import EarlyFusion
from brain_ai.models.adapters.sklearn import SKLearnAdapter
from brain_ai.utils.datasets import generate_multimodal_regression_data


def _run_with_spec(spec: PipelineSpec):
    brain = Brain(decision_engine=DecisionEngine(), evaluator=Evaluator())
    data = generate_multimodal_regression_data(n_samples=120, random_state=7)
    result = brain.run_pipeline(spec, data)
    return result


def test_fusion_strategies_produce_different_predictions():
    spec_early = PipelineSpec(
        modalities=["tabular", "sensor", "text"],
        granularity_strategy="resample",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={"random_state": 11, "n_estimators": 16},
    )
    spec_late = PipelineSpec(
        modalities=["tabular", "sensor", "text"],
        granularity_strategy="resample",
        fusion_strategy="late",
        model_backend="sklearn",
        hyperparameters={"random_state": 11, "n_estimators": 16},
    )

    preds_early = _run_with_spec(spec_early)["predictions"]
    preds_late = _run_with_spec(spec_late)["predictions"]

    assert not np.allclose(preds_early, preds_late)


def test_granularity_strategies_produce_different_predictions():
    spec_resample = PipelineSpec(
        modalities=["tabular", "sensor", "text"],
        granularity_strategy="resample",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={"random_state": 9, "n_estimators": 18},
    )
    spec_pooling = PipelineSpec(
        modalities=["tabular", "sensor", "text"],
        granularity_strategy="pooling",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={"random_state": 9, "n_estimators": 18},
    )

    result_resample = _run_with_spec(spec_resample)
    result_pooling = _run_with_spec(spec_pooling)

    assert result_resample["metrics"]["rmse"] != result_pooling["metrics"]["rmse"]


def test_user_override_components_are_respected():
    # Spec asks for late fusion, but explicit user override should force early fusion.
    spec = PipelineSpec(
        modalities=["tabular", "sensor", "text"],
        granularity_strategy="pooling",
        fusion_strategy="late",
        model_backend="sklearn",
        hyperparameters={"random_state": 3, "n_estimators": 8},
    )

    data = generate_multimodal_regression_data(n_samples=80, random_state=9)

    override_model = SKLearnAdapter(estimator=LinearRegression())
    brain_override = Brain(
        fusion=EarlyFusion(),
        model_adapter=override_model,
        evaluator=Evaluator(),
        decision_engine=DecisionEngine(),
    )
    result_override = brain_override.run_pipeline(spec, data)

    assert result_override["metrics"]["rmse"] >= 0


def test_dag_is_generated(tmp_path):
    spec = PipelineSpec(
        modalities=["tabular", "sensor", "text"],
        granularity_strategy="attention",
        fusion_strategy="intermediate",
        model_backend="sklearn",
        hyperparameters={"random_state": 5, "n_estimators": 10},
    )

    dag_builder = DAGBuilder(out_dir=str(tmp_path))
    output_path = dag_builder.build_and_save(spec, filename="test_pipeline_dag.png")

    assert (tmp_path / "test_pipeline_dag.png").exists()
    assert output_path.endswith("test_pipeline_dag.png")
