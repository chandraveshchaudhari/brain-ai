from brain_ai.core.brain import Brain
from brain_ai.decision.engine import DecisionEngine, generate_pipeline_combinations
from brain_ai.experiments.evaluator import Evaluator
from brain_ai.experiments.runner import Runner
from brain_ai.agents.executor import Executor
from brain_ai.utils.datasets import generate_multimodal_regression_data


def test_pipeline_combinations_run_without_error():
    specs = generate_pipeline_combinations(
        modalities=["tabular", "sensor", "text"],
        granularity_strategies=["resample", "pooling", "attention"],
        fusion_strategies=["early", "late", "intermediate"],
        model_backends=["sklearn"],
        hyperparameters={"random_state": 4, "n_estimators": 12},
    )

    brain = Brain(decision_engine=DecisionEngine(), evaluator=Evaluator())
    runner = Runner(executor=Executor(brain=brain))
    data = generate_multimodal_regression_data(n_samples=150, random_state=21)

    results = runner.run_batch(specs, data)

    assert len(results) == 9
    assert all("metrics" in result for result in results)
    assert all("predictions" in result for result in results)
