# Brain-AI (brain-automl): product requirements

Owner: Chandravesh Chaudhari (SolarpunkWorks Pvt Ltd). Updated 2026-10-26.

**What this file contains:** only features that already exist, verified in the code on
2026-10-26. New features are added here by the owner when he decides on them; the
Supervisor plans from this file and new app versions are released from it.

**Current goal (owner, 2026-10-26):** release what exists in working condition. That means
fixing what is broken, hardening security and making it faster. No new features until the
owner adds them below.

## Product in one line

An open-source engine that decides and executes multimodal ML pipelines: PipelineSpec = modalities, granularity, fusion, model backend and hyperparameters.

## Implemented features

1. **Pipeline decisions**: a PipelineSpec decision engine. (`src/brain_ai/decision`)
2. **Granularity alignment**: resample, pooling and attention strategies. (`src/brain_ai/granularity`)
3. **Fusion strategies**: early, intermediate and late fusion. (`src/brain_ai/fusion`)
4. **Model backends** through adapters. (`src/brain_ai/models`)
5. **Experiment DAG**: build and visualise experiment DAGs. (`src/brain_ai/dag`)
6. **Experiments**: runner, evaluator and leaderboard. (`src/brain_ai/experiments`)
7. **Reinforcement-learning search**: environment, state, action, reward, policy and trainer. (`src/brain_ai/rl`)
8. **Agents**: planner and executor with skills. (`src/brain_ai/agents`)
9. **Package**: published on PyPI as `brain-automl` (v0.0.7), with example notebooks in `examples/`.

## Release work: fix, secure, speed up

Each item is checkable. Fixing them is in scope now; adding features is not.

### Fix (current state)
- **Failing tests:** 6 tests in `tests/test_pipeline_behaviors.py` fail (e.g. granularity strategies, user-override components). Make them pass without weakening them.
- **README quick start:** runs as written on Python 3.10–3.12.
- **Gate green:** `bash scripts/test.sh` passes.

### Security
- No private data, keys or unpublished research results in this public repository.

### Speed
- The test suite runs in under 60 s (47 s today); the README example completes in under 30 s on the Mac.
