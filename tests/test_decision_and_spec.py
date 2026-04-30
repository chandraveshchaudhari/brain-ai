import json

from brain_ai.decision.engine import DecisionEngine, generate_pipeline_combinations
from brain_ai.decision.spec import PipelineSpec


def test_pipeline_spec_roundtrip_and_serialization():
    spec = PipelineSpec(
        modalities=["tabular", "sensor"],
        granularity_strategy="resample",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={"n_estimators": 10},
    )

    payload = spec.to_json()
    loaded = PipelineSpec.from_json(payload)

    assert json.loads(payload)["fusion_strategy"] == "early"
    assert loaded == spec


def test_generate_pipeline_combinations_cartesian_product():
    specs = generate_pipeline_combinations(
        modalities=["tabular", "sensor"],
        granularity_strategies=["resample", "pooling"],
        fusion_strategies=["early", "late", "intermediate"],
        model_backends=["sklearn"],
    )

    assert len(specs) == 2 * 3 * 1
    assert all(isinstance(spec, PipelineSpec) for spec in specs)


def test_decision_engine_resolves_components():
    engine = DecisionEngine()
    spec = PipelineSpec(
        modalities=["tabular"],
        granularity_strategy="resample",
        fusion_strategy="late",
        model_backend="sklearn",
        hyperparameters={},
    )

    components = engine.resolve_components(spec)

    assert "granularity" in components
    assert "fusion" in components
    assert "model_adapter" in components
