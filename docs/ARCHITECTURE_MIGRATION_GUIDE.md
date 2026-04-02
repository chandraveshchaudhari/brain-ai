"""Brain-AI Multimodal Framework: Architecture & Migration Guide

## Overview

The Brain-AI framework has been redesigned from a monolithic, modality-specific structure 
to a **plug-in-based, registry-driven architecture** that scales to any modality and any 
backend library. This document explains the new design and how to migrate existing code.

## Key Design Principles

### 1. **Protocol-Based Extensibility**
   - All backends implement `BaseLibraryBackend` (abstract interface)
   - All modality executors implement `BaseModalityExecutor`
   - New backends/modalities need only inherit the protocol class
   - No changes to core executor code required

### 2. **Auto-Discovery Registry**
   - Backends self-register on import (decorator pattern)
   - Executors query registry for available backends
   - LLM agents can discover tools/backends/modalities at runtime

### 3. **Standardized Outputs**
   - All backends return `ModalityResult` (normalized format)
   - Fusion layer receives consistent input regardless of backend
   - All metrics/metadata accessible in uniform structure

### 4. **Graceful Degradation**
   - Missing backends skip automatically (via `skip_unavailable_backends` config)
   - Partial results okay (run whatever backends are available)
   - No runtime errors if optional dependencies missing

### 5. **Privacy-First with Ollama Default**
   - Default LLM provider: Ollama (local, no data leak)
   - Cloud providers opt-in only
   - All configs support override

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│           LLM Agent Layer (Phase 4)                         │
│     Plan generation + Tool discovery + Execution           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│           Pipeline Runner (PipelineRunner)                  │
│     Execute ordered tool/modality chains with state         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│           Fusion Layer (BaseFusionStrategy)                 │
│     Combine multiple ModalityResults into one output        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│        Modality Executors (BaseModalityExecutor)            │
│   ┌──────────────┬──────────────┬──────────────┐            │
│   │ Tabular      │   Text       │ Time Series  │            │
│   │ Executor     │   Executor   │  Executor    │            │
│   └──────┬───────┴──────┬───────┴──────┬───────┘            │
└──────────┼──────────────┼──────────────┼───────────────────┘
           │              │              │
   ┌───────▼────────┬────▼────────┬────▼──────────┐
   │  BACKEND       │  BACKEND    │  BACKEND      │
   │  REGISTRY      │  REGISTRY   │  REGISTRY     │
   └────────────────┴─────────────┴───────────────┘
           │
   ┌───────▼──────────────────────────┐
   │ Individual Backend Implementations │
   │ • AutoGluon                        │
   │ • FLAML                            │
   │ • StatsForecast                    │
   │ • NeuralForecast                   │
   │ • PyCaret                          │
   │ • H2O                              │
   │ • (+ more)                         │
   └────────────────────────────────────┘
```

---

## Core Components

### 1. **Protocols** (`brain_automl/core/protocols.py`)

```python
from brain_automl.core import BaseLibraryBackend, BaseModalityExecutor

# Any backend must implement this:
class MyBackend(BaseLibraryBackend):
    name = "my_backend"
    modality = "time_series"
    task_types = ["forecasting", "anomaly_detection"]
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import my_library
            return True
        except ImportError:
            return False
    
    def fit(self, x_train, y_train, **kwargs):
        # real implementation
        pass
    
    def predict(self, model, x_test, **kwargs):
        # real implementation
        pass
```

### 2. **Registry** (`brain_automl/core/registry.py`)

```python
from brain_automl.core import BACKEND_REGISTRY

# Backends auto-register on import:
@BACKEND_REGISTRY.register()
class AutoGluonTimeSeriesBackend(BaseLibraryBackend):
    name = "autogluon_ts"
    # ...

# Later, query registry:
backends = BACKEND_REGISTRY.items()  # {'autogluon_ts': <class>, ...}
autogluon = BACKEND_REGISTRY.get("autogluon_ts")
available = BACKEND_REGISTRY.has("flaml")
```

### 3. **Standardized Results** (`brain_automl/core/result.py`)

```python
from brain_automl.core import ModalityResult, FusionResult

# All backends return this format:
result = ModalityResult(
    modality="time_series",
    backend="autogluon",
    task="forecasting",
    predictions=array([...]),           # numpy/pandas
    probabilities=array([...]),         # for classification tasks
    metrics={"mape": 0.05, "rmse": ...},  # e.g. forecasting metrics
    model_path="/tmp/model.pkl",          # optional serialization
    metadata={"feature_count": 10, ...}   # custom backend data
)
```

### 4. **Modality Executors** (e.g., `brain_automl/model_zoo/time_series_ai/`)

```python
from brain_automl.core import BaseModalityExecutor

class TimeSeriesAutoML(BaseModalityExecutor):
    modality = "time_series"
    
    def run(self, data, task="forecasting", backends=None, **kwargs):
        # 1. Load config
        config = get_default_config()
        
        # 2. Discover backends from registry
        # 3. Try each backend in order
        # 4. Return list of ModalityResult
        results = [...]
        return results
```

### 5. **Config Defaults** (`brain_automl/config/defaults.py`)

```python
from brain_automl.config import DEFAULT_CONFIG, get_default_config

config = get_default_config()
# {
#   "profile": "privacy_first",
#   "offline_mode": True,
#   "llm": {
#       "default_provider": "ollama",  # Privacy first!
#       "fallback_providers": [],      # No auto-fallback to cloud
#   },
#   "backends": {
#       "skip_unavailable_backends": True,  # Graceful degradation
#   },
#   "modalities": {
#       "enabled": ["tabular", "text", "time_series", ...],
#   },
#   "time_series": {
#       "default": ["autogluon", "statsforecast", "neuralforecast"],
#   },
#   ...
# }
```

---

## Migration Guide

### For New Code: Use the New Registry-Driven API

```python
# Modern way (recommended)
from brain_automl.model_zoo.time_series_ai import TimeSeriesAutoML

executor = TimeSeriesAutoML()
results = executor.run(data, task="forecasting")

for result in results:
    print(f"{result.backend}: {result.metrics}")  # ModalityResult format
```

### For Existing Code: Use the Legacy Bridge (No Changes!)

```python
# Old code still works unchanged:
from brain_automl import Brain

brain = Brain()
result = brain.run_time_series(data, timestamp_col="date", target_col="value")

# Result structure:
# {
#     "success": True/False,
#     "results": [
#         {"backend": "...", "predictions": ..., "metrics": ..., ...},
#         ...
#     ]
# }
```

**The legacy API now routes through the new registry internally. 
No code changes required, but new code should use TimeSeriesAutoML directly.**

---

## Implementation Checklist

### Completed ✅
- [x] Core protocols (BaseLibraryBackend, BaseModalityExecutor, etc.)
- [x] Registry with auto-discovery
- [x] Standardized result models (ModalityResult, FusionResult)
- [x] Config system with privacy-first defaults
- [x] Time-series executor scaffolding
- [x] 7 time-series backend stubs (AutoGluon, FLAML, StatsForecast, etc.)
- [x] Legacy Bridge (Brain class backward compatibility)
- [x] Core tests and backward compatibility tests
- [x] Example demo script

### In Progress ⏳
- [ ] Real backend implementations (fit/predict logic)
  - Priority: AutoGluon, StatsForecast
- [ ] Tabular modality refactor
- [ ] Text modality refactor
- [ ] Image/audio modalities
- [ ] Fusion strategies

### Planned 📋
- [ ] LLM Agent layer with tool discovery
- [ ] PipelineRunner integration with approval gates
- [ ] Benchmarking suite
- [ ] Data contracts
- [ ] CI/release gates
- [ ] Packaging & distribution

---

## Quick Start

### 1. Run the Demo
```bash
cd projects/core-research/brain-ai
python examples/demo_multimodal_framework.py
```

### 2. Run Tests
```bash
# Core architecture tests
python -m pytest tests/test_core_architecture.py -v

# Backward compatibility tests
python -m pytest tests/test_backward_compatibility.py -v
```

### 3. Use in Your Code

**Option A: New code (recommended)**
```python
from brain_automl.model_zoo.time_series_ai import TimeSeriesAutoML
from brain_automl.core import ModalityResult

executor = TimeSeriesAutoML()
results = executor.run(data)

for result in results:
    assert isinstance(result, ModalityResult)
    print(f"{result.backend}: {result.metrics}")
```

**Option B: Legacy code (still works)**
```python
from brain_automl import Brain

brain = Brain()
result = brain.run_time_series(data, timestamp_col="date", target_col="value")
print(f"Success: {result['success']}, Backends: {len(result['results'])}")
```

---

## File Structure

```
brain_automl/
├── __init__.py                  # Package exports (new + legacy)
├── core/                        # NEW: Core architecture layer
│   ├── __init__.py
│   ├── protocols.py             # ABC contracts
│   ├── registry.py              # Auto-discovery registries
│   ├── result.py                # ModalityResult, FusionResult
│   └── pipeline.py              # PipelineRunner orchestrator
│
├── config/                      # NEW: Concrete configuration
│   ├── __init__.py
│   └── defaults.py              # Privacy-first config dict
│
├── model_zoo/
│   ├── time_series_ai/          # NEW: Time-series modality
│   │   ├── __init__.py
│   │   ├── time_series_executor.py
│   │   ├── data_preparation.py
│   │   └── backends/
│   │       ├── autogluon_timeseries.py
│   │       ├── flaml_timeseries.py
│   │       ├── statsforecast_backend.py
│   │       ├── neuralforecast_backend.py
│   │       ├── pycaret_backend.py
│   │       ├── h2o_backend.py
│   │       └── optuna_backend.py
│   │
│   ├── tabular_data_ai/         # EXISTING
│   │   └── AutoML.py            # (will refactor in Phase 2)
│   │
│   └── text_data_ai/            # EXISTING
│       └── execution.py         # (will refactor in Phase 2)
│
├── legacy_bridge.py             # NEW: Brain backward compatibility
│
└── agent/                       # FUTURE: LLM agent layer
    ├── agent.py
    ├── tools/
    └── prompts/

tests/
├── __init__.py
├── test_core_architecture.py    # NEW: Registry, protocols, results
├── test_backward_compatibility.py  # NEW: Legacy Brain API
└── test_backends/               # FUTURE: Per-backend tests
```

---

## Key Takeaways

1. **Old API still works** → `from brain_automl import Brain` unchanged
2. **New API is cleaner** → `from brain_automl.model_zoo.time_series_ai import TimeSeriesAutoML`
3. **Extensibility via protocols** → Inherit `BaseLibraryBackend`, no core edits
4. **Auto-discovery via registry** → New backends register on import
5. **Standardized outputs** → All backends return `ModalityResult`
6. **Privacy first** → Ollama default, no cloud data leakage
7. **Graceful degradation** → Missing backends skip, no crashes

---

## Questions?

Refer to:
- Architecture plan: `docs/MULTIMODAL_ARCHITECTURE_PLAN.md`
- Backend examples: `src/brain_automl/model_zoo/time_series_ai/backends/`
- Tests: `tests/test_*.py`

"""
