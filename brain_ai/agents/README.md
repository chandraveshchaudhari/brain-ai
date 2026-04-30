Agents and Skills
=================

Purpose
-------
Provide a safe, modular layer for LLM-driven interactions. Skills accept
structured input and return structured output while calling internal APIs.

Available skills
----------------
- `generate_pipeline`: task_description -> `PipelineSpec`
- `run_experiment`: `PipelineSpec` + components -> metrics
- `compare_models`: list of results -> leaderboard
- `visualize_dag`: `PipelineSpec` + dag_builder -> image path
