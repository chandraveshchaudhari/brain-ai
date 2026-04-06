"""Test core architecture: registries, protocols, and config defaults."""

import pytest

from brain_automl.config import DEFAULT_CONFIG, get_default_config
from brain_automl.core import (
    BACKEND_REGISTRY,
    TOOL_REGISTRY,
    FusionResult,
    ModalityResult,
    PipelineRunner,
    PipelineStep,
)


class TestConfig:
    """Config defaults and retrieval."""

    def test_default_config_exists(self):
        """DEFAULT_CONFIG dict is non-empty."""
        assert isinstance(DEFAULT_CONFIG, dict)
        assert len(DEFAULT_CONFIG) > 0

    def test_default_config_ollama_first(self):
        """Default config specifies Ollama as default LLM provider."""
        assert DEFAULT_CONFIG["profile"] == "privacy_first"
        assert DEFAULT_CONFIG["llm"]["default_provider"] == "ollama"
        assert DEFAULT_CONFIG["offline_mode"] is True

    def test_get_default_config_returns_copy(self):
        """get_default_config returns a deep copy, not reference."""
        cfg1 = get_default_config()
        cfg2 = get_default_config()
        cfg1["profile"] = "modified"
        assert DEFAULT_CONFIG["profile"] == "privacy_first"
        assert cfg2["profile"] == "privacy_first"

    def test_time_series_optional_backends_include_chronos(self):
        """Chronos is exposed as an optional time-series backend."""
        optional = DEFAULT_CONFIG["backends"]["by_modality"]["time_series"]["optional"]
        assert "chronos" in optional


class TestRegistry:
    """Backend and tool registries."""

    def test_backend_registry_exists(self):
        """BACKEND_REGISTRY is available."""
        assert BACKEND_REGISTRY is not None
        assert hasattr(BACKEND_REGISTRY, "register")
        assert hasattr(BACKEND_REGISTRY, "get")
        assert hasattr(BACKEND_REGISTRY, "has")

    def test_tool_registry_exists(self):
        """TOOL_REGISTRY is available."""
        assert TOOL_REGISTRY is not None
        assert hasattr(TOOL_REGISTRY, "register")


class TestResult:
    """Result data models."""

    def test_modality_result_creation(self):
        """ModalityResult can be instantiated."""
        result = ModalityResult(
            modality="test_modality",
            backend="test_backend",
            task="test_task",
            predictions=[1, 2, 3],
        )
        assert result.modality == "test_modality"
        assert result.backend == "test_backend"
        assert result.task == "test_task"
        assert result.predictions == [1, 2, 3]

    def test_fusion_result_creation(self):
        """FusionResult can be instantiated."""
        result = FusionResult(
            strategy="decision_fusion",
            predictions=[1, 2, 3],
        )
        assert result.strategy == "decision_fusion"
        assert result.predictions == [1, 2, 3]


class TestPipelineRunner:
    """Pipeline execution utilities."""

    def test_pipeline_runner_runs_steps(self):
        """PipelineRunner executes steps in order."""

        def step1(**kwargs):
            return {"step": 1}

        def step2(**kwargs):
            return {"step": 2}

        runner = PipelineRunner()
        steps = [
            PipelineStep("step1", step1),
            PipelineStep("step2", step2),
        ]
        outputs = runner.run(steps)
        assert len(outputs) == 2
        assert outputs[0]["step"] == 1
        assert outputs[1]["step"] == 2
        assert len(runner.history) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
