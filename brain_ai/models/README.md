Models module
=============

Purpose
-------
Provide thin adapters to model backends (sklearn, AutoGluon, TPOT, ...).

API
---
- `BaseModelAdapter`: `fit(X, y)`, `predict(X)`

Adapters must avoid business logic and translate between framework APIs and
the system interface.
