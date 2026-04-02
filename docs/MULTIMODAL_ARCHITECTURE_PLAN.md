# Plan: Brain-AI Architecture Redesign + Full Modality Support + LLM Agent Layer

## Status: Draft — awaiting user approval

## Context
- Codebase: `/Volumes/MacSSD/01_Projects/Chandravesh-ML-Research/projects/core-research/brain-ai/src/brain_automl/`
- Key file: `model_zoo/tabular_data_ai/AutoML.py` — all AutoML wrappers in one class
- Sentiment: `model_zoo/text_data_ai/execution.py` — FinBERT + Twitter RoBERTa
- Deps: `setup.py` with pinned old versions (~2023)
- Docs: `docs/01-05_*.md` — vision, architecture, roadmap

## Version Change Summary
| Library | Current | Latest (Apr 2026) | Breaking? |
|---|---|---|---|
| autogluon | 0.8.3b | ~1.2.x | Partial (feature gen imports) |
| autokeras | 1.1.0 | 3.0.0 | YES — Keras 3 rewrite |
| auto-sklearn | 0.15.0 | 0.15.x (discontinued) | Needs Linux note |
| TPOT | 0.12.0 | 0.12.x / TPOT2 | Partial |
| pycaret | 3.1.0 | 3.3.2 | Requires Python 3.9+ |
| h2o | latest | latest | API stable |
| mljar-supervised | 0.11.5 | ~1.x | Possible API changes |
| transformers | 4.31.0 | ~4.40+ | Minor |
| tensorflow | 2.11.0 | 2.16+ | Moderate |
| torch | 1.13.1+cpu | 2.x | API stable |

## Phase 1: Dependency Audit & Update — setup.py / requirements
1. Update `setup.py` with latest pinned versions for each library
2. Add Python version constraint (>=3.10 for full compat)
3. Note auto-sklearn limitation (Linux/macOS, Python <3.11)
4. Replace `mxnet` dep (dropped by AutoGluon 1.x)
5. Move to `pyproject.toml` optionally (nice-to-have)

## Phase 2: Fix AutoML.py — Breaking Import Changes
1. **AutoGluon**: Update feature generator imports (paths moved in 1.x)
   - Old: `from autogluon.common.features.types import R_INT, R_FLOAT`
   - Check: `from autogluon.features.generators import ...` paths
2. **AutoKeras**: Major rewrite — 3.0.0 uses Keras 3 multi-backend
   - Old API: `ak.StructuredDataClassifier(...)` with TF backend
   - New API: Same surface but needs `keras>=3.0.0`, backend config
   - Add backend env var: `os.environ["KERAS_BACKEND"] = "tensorflow"` (or jax/torch)
3. **PyCaret**: 3.1 → 3.3.2 — update `pycaret.classification` setUp API differences
4. **TPOT**: If staying with 0.12.x, minimal changes. If TPOT2, add note.
5. **MLJar**: Check for API changes in `supervised.AutoML` class name/params
6. **H2O**: Generally stable, check init/connect API

## Phase 3: New Time Series Module
Create `src/brain_automl/model_zoo/time_series_ai/` with:
1. `__init__.py`
2. `time_series_executor.py` — Main class `TimeSeriesAutoML`
   - `autogluon_timeseries()` — AutoGluon TimeSeriesPredictor (best maintained)
   - `pycaret_timeseries()` — PyCaret TSForecastingExperiment
   - `h2o_timeseries()` — H2O AutoML with time series features
3. `data_preparation.py` — Helper for converting DataFrames to time series format
   - AutoGluon format: `item_id`, `timestamp`, `target` columns
   - PyCaret format: date index with target column

## Phase 4: Test Script / Notebook
1. Create `notebooks/time_series_test.ipynb` under `model_zoo/time_series_ai/`
2. Use sample dataset (e.g., airline passengers, AirQuality, or synthetic sine wave)
3. Run AutoGluon TimeSeriesPredictor with 5-minute time limit
4. Run PyCaret TSForecastingExperiment
5. Compare forecasts visually

## Relevant Files
- `src/brain_automl/model_zoo/tabular_data_ai/AutoML.py` — Patch imports, update AutoKeras/PyCaret/MLJar
- `src/brain_automl/model_zoo/text_data_ai/execution.py` — Update transformers API (minor)
- `setup.py` or `pyproject.toml` — Update all version pins
- NEW: `src/brain_automl/model_zoo/time_series_ai/__init__.py`
- NEW: `src/brain_automl/model_zoo/time_series_ai/time_series_executor.py`
- NEW: `src/brain_automl/model_zoo/time_series_ai/data_preparation.py`
- NEW: `notebooks/time_series_test.ipynb` (or `.py` script)

## Verification
1. `pip install -e .` runs without dependency conflicts
2. `import brain_automl` loads without error
3. AutoGluon, PyCaret, H2O time series methods run on sample data
4. `time_series_test.ipynb` executes end-to-end

## Decisions / Exclusions
- Auto-sklearn: Keep but document it's Linux-only + Python <3.11
- AutoKeras 3.0: Keep but note it now supports TF/JAX/PyTorch backends
- Sentiment analysis (BERT/RoBERTa): Minor transformers update only, no redesign
- Multimodal fusion: Out of scope for this phase
- TPOT2: Note as optional upgrade path, not required now
- FLAML: Could be added as new library, but not in scope unless user wants it
