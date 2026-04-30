import importlib

import numpy as np
import pytest

from src.brain_ai.agents.planner import Planner
from src.brain_ai.agents.skills.compare_models import compare_models
from src.brain_ai.agents.skills.generate_pipeline import generate_pipeline
from src.brain_ai.agents.skills.run_experiment import run_experiment
from src.brain_ai.agents.skills.visualize_dag import visualize_dag
from src.brain_ai.core.brain import Brain
from src.brain_ai.dag.builder import DAGBuilder
from src.brain_ai.decision.engine import DecisionEngine
from src.brain_ai.decision.spec import PipelineSpec
from src.brain_ai.experiments.evaluator import Evaluator
from src.brain_ai.experiments.leaderboard import Leaderboard
from src.brain_ai.fusion.base import BaseFusion
from src.brain_ai.fusion.intermediate import IntermediateFusion
from src.brain_ai.granularity.base import BaseGranularity
from src.brain_ai.granularity.pooling import PoolingGranularity
from src.brain_ai.granularity.resample import ResampleGranularity
from src.brain_ai.models.adapters.autogluon import AutoGluonAdapter
from src.brain_ai.models.adapters.tpot import TPOTAdapter
from src.brain_ai.models.base import BaseModelAdapter
from src.brain_ai.rl.action import Action
from src.brain_ai.rl.config import DEFAULT
from src.brain_ai.rl.environment import Environment
from src.brain_ai.rl.policy import RandomPolicy
from src.brain_ai.rl.reward import RewardConfig, RewardFunction, compute_reward
from src.brain_ai.rl.state import State
from src.brain_ai.rl.trainer import Trainer
from src.brain_ai.utils import ensure_list
from src.brain_ai.utils.datasets import generate_multimodal_regression_data


def test_planner_and_skills_structured_outputs(tmp_path):
    planner = Planner()
    spec = planner.plan("tabular regression")
    assert spec.model_backend == "sklearn"

    generated = generate_pipeline("demo")
    assert generated["status"] == "ok"
    assert generated["pipeline_spec"]["fusion_strategy"] == "early"

    data = generate_multimodal_regression_data(n_samples=30, random_state=5)
    engine = DecisionEngine()
    brain = Brain(decision_engine=engine, evaluator=Evaluator())
    run_payload = run_experiment(spec, {"brain": brain}, data)
    assert run_payload["status"] == "ok"

    cmp_payload = compare_models([run_payload])
    assert cmp_payload["status"] == "ok"
    assert len(cmp_payload["leaderboard"]) == 1

    dag_payload = visualize_dag(spec, DAGBuilder(out_dir=str(tmp_path)), out_path="skills_dag.png")
    assert dag_payload["status"] == "ok"
    assert dag_payload["dag_path"].endswith("skills_dag.png")

    # Ensure package exports are covered.
    skills_module = importlib.import_module("brain_ai.agents.skills")
    assert "generate_pipeline" in skills_module.__all__

    # Cover run_experiment branch that constructs Brain from components.
    class TinyGranularity:
        def align(self, raw_data):
            return raw_data

    class TinyFusion:
        def fuse(self, aligned):
            return {"X": np.asarray(aligned["modalities"]["tabular"]), "y": np.asarray(aligned["y"])}

    class TinyModel:
        def fit(self, X, y):
            self.val = float(np.mean(y))

        def predict(self, X):
            return np.full(len(X), self.val)

    run_payload_2 = run_experiment(
        spec,
        {
            "granularity": TinyGranularity(),
            "fusion": TinyFusion(),
            "model_adapter": TinyModel(),
            "evaluator": Evaluator(),
        },
        data,
    )
    assert run_payload_2["status"] == "ok"


def test_brain_error_paths_and_component_resolution():
    spec = PipelineSpec(
        modalities=["tabular"],
        granularity_strategy="resample",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={"random_state": 1, "n_estimators": 5},
    )
    data = generate_multimodal_regression_data(n_samples=20, random_state=1)

    with pytest.raises(RuntimeError):
        Brain().run_pipeline(spec, data)

    class DummyGranularity:
        def align(self, raw_data):
            return raw_data

    with pytest.raises(RuntimeError):
        Brain(granularity=DummyGranularity()).run_pipeline(spec, data)

    class DummyFusion:
        def fuse(self, aligned):
            return {"X": np.asarray(aligned["modalities"]["tabular"]), "y": np.asarray(aligned["y"])}

    with pytest.raises(RuntimeError):
        Brain(granularity=DummyGranularity(), fusion=DummyFusion()).run_pipeline(spec, data)

    class DirectModel:
        def fit(self, X, y):
            self._mean = float(np.mean(y))

        def predict(self, X):
            return np.full(shape=(len(X),), fill_value=self._mean)

    brain = Brain(
        granularity=DummyGranularity(),
        fusion=DummyFusion(),
        model_adapter=DirectModel(),
        evaluator=Evaluator(),
    )
    output = brain.run_pipeline(spec.to_dict(), data)
    assert "predictions" in output
    assert "score" in output["metrics"]

    assert brain._resolve_component("x", None, "k") == "x"
    assert brain._resolve_component(None, None, "k") is None


def test_brain_mlflow_and_decision_engine_runtime_branch(monkeypatch, tmp_path):
    brain_module = importlib.import_module("brain_ai.core.brain")

    class MlflowStub:
        def __init__(self):
            self.metrics = None
            self.artifact_path = None

        def log_artifact(self, path, artifact_path=None):
            self.artifact_path = (path, artifact_path)

        def log_metrics(self, metrics):
            self.metrics = metrics

    mlflow_stub = MlflowStub()
    monkeypatch.setattr(brain_module, "mlflow", mlflow_stub)

    spec = PipelineSpec(["tabular"], "resample", "early", "sklearn", {"n_estimators": 5, "random_state": 1})
    data = generate_multimodal_regression_data(n_samples=24, random_state=12)

    brain = Brain(
        decision_engine=DecisionEngine(),
        evaluator=Evaluator(),
        dag_builder=DAGBuilder(out_dir=str(tmp_path)),
    )
    result = brain.run_pipeline(spec, data)
    assert "predictions" in result
    assert mlflow_stub.metrics is not None
    assert mlflow_stub.artifact_path is not None


def test_spec_mlflow_and_decision_engine_error_paths(monkeypatch):
    spec = PipelineSpec(
        modalities=["tabular", "sensor"],
        granularity_strategy="resample",
        fusion_strategy="early",
        model_backend="sklearn",
        hyperparameters={"a": 1},
    )

    class MlflowStub:
        def __init__(self):
            self.params = {}

        def log_params(self, params):
            self.params.update(params)

        def log_param(self, key, value):
            self.params[key] = value

    stub = MlflowStub()
    spec_module = importlib.import_module("brain_ai.decision.spec")
    monkeypatch.setattr(spec_module, "mlflow", stub)
    spec.log_mlflow()
    assert "pipeline_spec_json" in stub.params
    assert "hyperparameters" in stub.params

    # cover mlflow-is-None branch
    monkeypatch.setattr(spec_module, "mlflow", None)
    spec.log_mlflow()

    engine = DecisionEngine()
    bad_granularity = PipelineSpec(["tabular"], "unknown", "early", "sklearn", {})
    bad_fusion = PipelineSpec(["tabular"], "resample", "unknown", "sklearn", {})
    bad_model = PipelineSpec(["tabular"], "resample", "early", "unknown", {})

    with pytest.raises(KeyError):
        engine.resolve_components(bad_granularity)
    with pytest.raises(KeyError):
        engine.resolve_components(bad_fusion)
    with pytest.raises(KeyError):
        engine.resolve_components(bad_model)

    auto = engine._build_model_adapter(AutoGluonAdapter, {"predictor": object()})
    tpot = engine._build_model_adapter(TPOTAdapter, {"tpot_estimator": object()})
    assert isinstance(auto, AutoGluonAdapter)
    assert isinstance(tpot, TPOTAdapter)

    class DummyAdapter:
        def __init__(self):
            self.ok = True

    fallback = engine._build_model_adapter(DummyAdapter, {})
    assert isinstance(fallback, DummyAdapter)


def test_dag_optional_paths_and_visualizer_error(monkeypatch, tmp_path):
    builder_module = importlib.import_module("brain_ai.dag.builder")
    visualizer_module = importlib.import_module("brain_ai.dag.visualizer")

    spec = PipelineSpec(["tabular"], "resample", "early", "sklearn", {})
    builder = DAGBuilder(out_dir=str(tmp_path))

    # branch where nx is unavailable
    monkeypatch.setattr(builder_module, "nx", None)
    assert builder.build(spec) is None
    with pytest.raises(RuntimeError):
        builder.build_and_save(spec)

    # branch where plot backends are unavailable
    monkeypatch.setattr(visualizer_module, "plt", None)
    with pytest.raises(RuntimeError):
        visualizer_module.save_graph_png(object(), str(tmp_path / "x.png"))


def test_evaluator_leaderboard_base_interfaces_and_utils():
    ev = Evaluator()
    score_only = ev.evaluate(np.array([1.0, 2.0]), None)
    assert "score" in score_only

    lb = Leaderboard()
    lb.add({"metrics": {"score": 1.0}})
    assert lb.summary()[0]["metrics"]["score"] == 1.0

    # Abstract method lines raising NotImplementedError
    with pytest.raises(NotImplementedError):
        BaseFusion.fuse(object(), None)
    with pytest.raises(NotImplementedError):
        BaseGranularity.align(object(), None)
    with pytest.raises(NotImplementedError):
        BaseModelAdapter.fit(object(), None, None)
    with pytest.raises(NotImplementedError):
        BaseModelAdapter.predict(object(), None)

    data = generate_multimodal_regression_data(n_samples=8, random_state=0)
    pooled_max = PoolingGranularity(window=2, method="max").align(data)
    assert pooled_max["modalities"]["tabular"].shape[0] > 0

    # hit chunk.size == 0 branch
    zero_width = {"modalities": {"tabular": np.empty((2, 0))}, "y": np.array([1.0, 2.0])}
    pooled_zero_width = PoolingGranularity(window=2, method="mean").align(zero_width)
    assert pooled_zero_width["modalities"]["tabular"].size == 0

    # hit empty sampled branch in resample
    empty = {"modalities": {"tabular": np.empty((0, 3))}, "y": np.empty((0,))}
    sampled = ResampleGranularity(step=2).align(empty)
    assert sampled["modalities"]["tabular"].shape[0] == 0

    # hit interaction-none branch for single modality
    fused = IntermediateFusion().fuse({"modalities": {"tabular": np.ones((5, 3))}, "y": np.ones(5)})
    assert fused["X"].shape[1] == 4

    assert ensure_list(None) == []
    assert ensure_list([1]) == [1]
    assert ensure_list("x") == ["x"]


def test_model_adapter_and_rl_scaffolding():
    with pytest.raises(RuntimeError):
        AutoGluonAdapter().fit([[1.0]], [1.0])
    with pytest.raises(RuntimeError):
        TPOTAdapter().fit([[1.0]], [1.0])

    class Predictor:
        def fit(self, X, y):
            self.fitted = True

        def predict(self, X):
            return np.zeros(len(X))

    class Estimator:
        def fit(self, X, y):
            self.fitted = True

        def predict(self, X):
            return np.ones(len(X))

    ag = AutoGluonAdapter(Predictor())
    ag.fit([[1], [2]], [1, 1])
    assert ag.predict([[1], [2]]).shape[0] == 2

    tp = TPOTAdapter(Estimator())
    tp.fit([[1], [2], [3]], [0, 1, 0])
    assert tp.predict([[1], [2], [3]]).shape[0] == 3

    assert compute_reward(1.0, 2.0, 3.0) == pytest.approx(0.95)
    rf = RewardFunction(RewardConfig(lambda_time=0.5, lambda_complexity=0.25))
    assert rf(4.0, 2.0, 4.0) == pytest.approx(2.0)

    env = Environment(dataset_metadata={"n": 1})
    state = env.reset()
    assert isinstance(state, State)
    assert state.metadata["n"] == 1

    policy = RandomPolicy(action_space={"granularity": ["resample"], "fusion": ["early"], "model": ["sklearn"]})
    action = policy.select(state)
    assert isinstance(action, Action)

    trainer = Trainer(env, policy)
    traces = trainer.train(episodes=2)
    assert len(traces) == 2

    assert DEFAULT["lam_time"] == 0.01

    rl_module = importlib.import_module("brain_ai.rl")
    assert "Action" in rl_module.__all__
