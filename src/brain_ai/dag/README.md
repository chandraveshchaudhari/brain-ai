DAG module
==========

Purpose
-------
Build and visualize pipeline DAGs. Uses `networkx` to create graphs and
`matplotlib` to export PNG images.

API
---
- `DAGBuilder.build(spec) -> networkx.DiGraph`
- `DAGBuilder.build_and_save(spec, filename) -> path`
