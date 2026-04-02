"""Tests for legacy Brain backward compatibility."""

import pandas as pd
import numpy as np

from brain_automl import Brain, LegacyBrainBridge
from brain_automl.core import ModalityResult


def test_legacy_brain_import():
    """Verify that Brain can be imported from brain_automl package."""
    assert Brain is not None
    assert LegacyBrainBridge is not None


def test_legacy_brain_instantiation():
    """Verify that Brain() can be instantiated like old code."""
    brain = Brain()
    assert brain is not None
    assert hasattr(brain, "run_time_series")
    assert hasattr(brain, "list_available_backends")


def test_legacy_brain_list_backends():
    """Verify that list_available_backends returns proper dict."""
    brain = Brain()
    backends = brain.list_available_backends()
    
    assert isinstance(backends, dict)
    # At least some backends should be registered (even if not installed)
    assert len(backends) > 0
    # Values should be booleans indicating availability
    for name, available in backends.items():
        assert isinstance(name, str)
        assert isinstance(available, bool)


def test_legacy_brain_with_synthetic_data():
    """Verify that legacy Brain can process synthetic time-series data."""
    brain = Brain()
    
    # Create simple time-series data
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "value": np.sin(np.arange(50) * 0.1) + np.random.normal(0, 0.1, 50),
    })
    
    # Call legacy API
    result = brain.run_time_series(
        data,
        timestamp_col="date",
        target_col="value",
        task="forecasting"
    )
    
    # Check structure
    assert isinstance(result, dict)
    assert "success" in result
    assert "results" in result
    assert isinstance(result["results"], list)


def test_legacy_brain_config_passthrough():
    """Verify that config is accepted (even if not fully used yet)."""
    config = {"test_key": "test_value"}
    brain = Brain(config=config)
    
    assert brain.config == config
    # Should not raise an error


def test_backward_compat_missing_columns():
    """Verify that proper errors are raised for missing columns."""
    brain = Brain()
    
    data = pd.DataFrame({"wrong_col": [1, 2, 3]})
    
    # Should raise ValueError for missing timestamp column
    try:
        brain.run_time_series(
            data,
            timestamp_col="missing_date",
            target_col="missing_value"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "missing_date" in str(e) or "not found" in str(e)


if __name__ == "__main__":
    print("Running backward compatibility tests...")
    test_legacy_brain_import()
    print("✓ test_legacy_brain_import")
    
    test_legacy_brain_instantiation()
    print("✓ test_legacy_brain_instantiation")
    
    test_legacy_brain_list_backends()
    print("✓ test_legacy_brain_list_backends")
    
    test_legacy_brain_with_synthetic_data()
    print("✓ test_legacy_brain_with_synthetic_data")
    
    test_legacy_brain_config_passthrough()
    print("✓ test_legacy_brain_config_passthrough")
    
    test_backward_compat_missing_columns()
    print("✓ test_backward_compat_missing_columns")
    
    print("\nAll backward compatibility tests passed!")
