"""Concrete default configuration used for new Brain-AI architecture."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "profile": "privacy_first",
    "offline_mode": True,
    "allow_cloud_fallback": False,
    "environment": "local_dev",
    "llm": {
        "default_provider": "ollama",
        "provider_priority": ["ollama", "openai", "anthropic"],
        "fallback_policy": "manual_approval_required",
        "ollama": {
            "base_url": "http://localhost:11434",
            "default_model": "qwen2.5-coder:14b",
            "planner_model": "qwen2.5:14b",
            "tool_model": "qwen2.5-coder:14b",
            "max_context_tokens": 32768,
            "temperature": 0.1,
            "top_p": 0.9,
        },
    },
    "agent": {
        "mode": "plan_then_execute",
        "require_user_approval_before_run": True,
        "max_retries_per_step": 2,
        "max_total_tool_calls": 30,
        "tool_timeout_seconds": 600,
        "persist_run_artifacts": True,
        "persist_plans": True,
        "fail_on_missing_schema": True,
    },
    "tools": {
        "autodiscovery_enabled": True,
        "discovery_paths": ["brain_automl/agent/tools", "brain_automl/modalities"],
        "require_json_schema": True,
        "allow_unsafe_tools": False,
        "audit_log_enabled": True,
    },
    "modalities": {
        "enabled": ["tabular", "text", "time_series", "image", "audio", "multimodal"],
        "required_minimum_for_fusion": 2,
        "auto_detect_from_dataset": True,
        "allow_partial_success": True,
    },
    "backends": {
        "by_modality": {
            "tabular": {
                "default": ["autogluon", "pycaret", "flaml"],
                "optional": ["autosklearn", "tpot", "h2o", "mljar", "autokeras"],
            },
            "text": {
                "default": ["huggingface", "finbert", "roberta"],
            },
            "time_series": {
                "default": ["autogluon_timeseries", "statsforecast", "neuralforecast", "flaml"],
                "optional": ["pycaret_timeseries", "h2o_timeseries", "optuna_tuner"],
            },
            "image": {
                "default": ["huggingface_vision"],
            },
            "audio": {
                "default": ["huggingface_audio"],
            },
            "multimodal": {
                "default": ["autogluon_multimodal"],
            },
        },
        "check_import_availability_on_startup": True,
        "skip_unavailable_backends": True,
        "report_unavailable_backends_in_summary": True,
        "hard_fail_if_no_backend_available_for_modality": True,
    },
    "data": {
        "default_test_size": 0.2,
        "random_seed": 42,
        "time_series": {
            "default_prediction_length": 14,
            "default_frequency": "auto",
        },
        "categorical_encoding_mode": "auto",
        "missing_value_strategy": "auto",
    },
    "fusion": {
        "enabled": False,
        "default_strategy": "decision_fusion",
        "decision": {
            "method": "weighted_average",
            "weights_mode": "metric_based",
            "metric_for_weights": "validation_score",
        },
        "min_modalities_required": 2,
    },
    "evaluation": {
        "primary_metric_by_task": {
            "classification": "f1_weighted",
            "regression": "rmse",
            "forecasting": "mase",
        },
        "secondary_metrics_enabled": True,
        "save_predictions": True,
        "generate_model_cards": True,
    },
    "logging": {
        "level": "INFO",
        "json_logs": True,
        "redact_sensitive_fields": True,
    },
    "reproducibility": {
        "save_config_snapshot_per_run": True,
        "save_library_versions": True,
        "save_random_seeds": True,
    },
    "privacy": {
        "default_data_residency": "local_only",
        "allow_external_api_calls": False,
        "require_explicit_opt_in_for_cloud": True,
        "block_upload_of_raw_datasets": True,
    },
    "developer": {
        "vscode_first_workflow": True,
        "copilot_assisted_mode": "optional",
        "require_manual_review_for_agent_generated_code": True,
    },
    "compat": {
        "enable_legacy_brain_class": True,
        "enable_legacy_model_zoo_paths": True,
        "deprecation_warning_level": "INFO",
        "target_removal_phase": "after_multimodal_stabilization",
    },
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep-copied default config so callers can mutate safely."""
    return deepcopy(DEFAULT_CONFIG)
