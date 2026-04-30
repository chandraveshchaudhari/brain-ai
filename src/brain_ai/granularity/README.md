Granularity module
==================

Purpose
-------
Provides strategies for aligning multimodal inputs in time (resampling,
pooling, attention-based alignment).

API
---
- `BaseGranularity`: `align(data) -> aligned_data`

Extend
------
Add new strategies under this package that inherit from `BaseGranularity`.
