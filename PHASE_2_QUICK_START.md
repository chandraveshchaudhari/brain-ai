"""Quick Reference: Phase 2 Backend Implementation

## One-Minute Overview

**Goal**: Fill in the 7 backend stubs with real fit/predict implementations

**Status**: Scaffolding complete ✅ → Now just needs implementation 🔧

**Effort**: ~3-4 days for all 7 backends (1 backend ≈ 4-6 hours)

---

## The Pattern (Copy This)

Each backend file follows the same pattern. Example: AutoGluon

```python
# File: src/brain_automl/model_zoo/time_series_ai/backends/autogluon_timeseries.py

from brain_automl.core import BaseLibraryBackend, BACKEND_REGISTRY
from brain_automl.model_zoo.time_series_ai.data_preparation import to_autogluon_format


@BACKEND_REGISTRY.register()
class AutoGluonTimeSeriesBackend(BaseLibraryBackend):
    name = "autogluon"
    modality = "time_series"
    task_types = ("forecasting", "anomaly_detection")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if AutoGluon is installed."""
        try:
            import autogluon.timeseries as ats
            return True
        except ImportError:
            return False
    
    def fit(self, x_train, y_train, **kwargs):
        """Train AutoGluon TimeSeriesPredictor.
        
        Args:
            x_train: DataFrame with features + timestamp
            y_train: DataFrame with target values
            **kwargs: Backend-specific hyperparams (e.g., time_limit)
        
        Returns:
            Trained model object (or handle to serialized model)
        """
        import autogluon.timeseries as ats
        
        # Convert to AutoGluon format
        train_data = to_autogluon_format(x_train, y_train)
        
        # Initialize predictor
        predictor = ats.TimeSeriesPredictor(
            prediction_length=kwargs.get("forecast_horizon", 12),
            freq=kwargs.get("freq", "D"),
            target=kwargs.get("target_col", "target"),
            time_limit=kwargs.get("time_limit", 600),
        )
        
        # Train
        predictor.fit(train_data)
        
        return predictor
    
    def predict(self, model, x_test, **kwargs):
        """Generate predictions.
        
        Args:
            model: Trained predictor object
            x_test: DataFrame with test features
            **kwargs: Backend-specific params
        
        Returns:
            dict with 'predictions' (numpy array) and 'probabilities' (None)
        """
        # Generate predictions
        predictions = model.predict(x_test)
        
        # Convert to expected format
        return {
            "predictions": predictions.values,
            "probabilities": None,  # Not applicable for regression
            "metrics": {},  # Optional, computed by executor
        }
```

**That's the template!** Just:
1. Import your library
2. Implement `fit()` with real training logic
3. Implement `predict()` with real prediction logic
4. Return standardized dict format

---

## Priority Order

### Must Do First (Foundational)
1. **AutoGluon** (most mature, most tested)
2. **StatsForecast** (simplest API, fast)

### Quick Wins (Simple APIs)
3. **NeuralForecast** (Nixtla ecosystem, similar to StatsForecast)
4. **PyCaret** (high-level API)

### Medium Effort
5. **FLAML** (hyperparameter optimization)
6. **H2O** (distributed training)

### Complex (Save for Later)
7. **Optuna** (pure hyperparameter tuner, different use case)

---

## Checklist for Each Backend

For each backend file, verify:

- [ ] Import library at top of file
- [ ] is_available() checks import (returns True/False)
- [ ] fit() accepts x_train, y_train, **kwargs
- [ ] fit() returns model object
- [ ] predict() accepts model, x_test, **kwargs
- [ ] predict() returns dict with 'predictions' and 'probabilities' keys
- [ ] Code has docstrings
- [ ] Type hints on all methods
- [ ] @BACKEND_REGISTRY.register() decorator present
- [ ] class name follows pattern {Library}Backend or {Library}TimeSeriesBackend

---

## Testing Each Backend

```bash
# Create a simple test file: tests/test_backends_autogluon.py

import pandas as pd
import numpy as np
import pytest
from brain_automl.model_zoo.time_series_ai.backends.autogluon_timeseries import (
    AutoGluonTimeSeriesBackend
)


def test_autogluon_available():
    """Check if AutoGluon is installed."""
    # Skip if not available
    if not AutoGluonTimeSeriesBackend.is_available():
        pytest.skip("AutoGluon not installed")


def test_autogluon_fit_predict():
    """Test fit/predict cycle."""
    if not AutoGluonTimeSeriesBackend.is_available():
        pytest.skip("AutoGluon not installed")
    
    backend = AutoGluonTimeSeriesBackend()
    
    # Create sample data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    x_train = pd.DataFrame({
        "date": dates,
        "feature": np.random.randn(100),
    })
    y_train = pd.DataFrame({
        "target": np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
    })
    
    # Fit
    model = backend.fit(x_train, y_train, forecast_horizon=10)
    assert model is not None
    
    # Prepare test data
    x_test = pd.DataFrame({
        "date": pd.date_range("2023-04-11", periods=10, freq="D"),
        "feature": np.random.randn(10),
    })
    
    # Predict
    result = backend.predict(model, x_test)
    assert "predictions" in result
    assert len(result["predictions"]) > 0
```

Run with:
```bash
python -m pytest tests/test_backends_autogluon.py -v
```

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Wrong input format | Use data_preparation.py helpers (to_autogluon_format, etc.) |
| Library not installed | is_available() protects with try/except |
| Predictions not array | Convert with .values or np.array() |
| Wrong output dict keys | Must have 'predictions' and 'probabilities' |
| kwargs ignored | Always accept **kwargs even if unused |
| Model not serializable | Optional (saves to model_path later) |

---

## Helper Functions Available

In `data_preparation.py`, use these to convert formats:

```python
from brain_automl.model_zoo.time_series_ai.data_preparation import (
    to_autogluon_format,
    to_pycaret_format,
    to_neuralforecast_format,
)

# These handle column renaming, index setting, etc. for each library
```

---

## How to Debug

```bash
# If backend fails, check:

# 1. Is library installed?
python -c "import autogluon; print(autogluon.__version__)"

# 2. Does is_available() work?
python -c "from brain_automl.model_zoo.time_series_ai.backends.autogluon_timeseries import AutoGluonTimeSeriesBackend; print(AutoGluonTimeSeriesBackend.is_available())"

# 3. Can you import the backend?
python -c "from brain_automl.model_zoo.time_series_ai import backends; print(backends)"

# 4. Is it registered?
python -c "from brain_automl.core import BACKEND_REGISTRY; print(BACKEND_REGISTRY.items())"
```

---

## Integration Points

Once a backend is implemented:

1. Run demo to see it in action:
   ```bash
   python examples/demo_multimodal_framework.py
   ```

2. It will automatically appear in BACKEND_REGISTRY
3. TimeSeriesAutoML executor will discover it
4. Tests will validate input/output

**No other files to touch!**

---

## Estimated Timeline

- AutoGluon: 4-6 hours (most documented)
- StatsForecast: 3-4 hours (simple API)
- NeuralForecast: 3-4 hours (similar to StatsForecast)
- PyCaret: 3-4 hours (high-level)
- FLAML: 4-5 hours (tuning logic)
- H2O: 4-5 hours (distributed)
- Optuna: 3-4 hours (hyperparameter tuner)

**Parallelizable**: Can assign different backends to different people

---

## Next Steps (Immediate)

1. Read the template above
2. Open one backend file (start with AutoGluon or StatsForecast)
3. Replace the `pass` statements with real implementation
4. Test it with the test template above
5. Run `git status` to see what changed
6. Repeat for next backend

**That's it!** 🚀
"""
