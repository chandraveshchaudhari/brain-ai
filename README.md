
# Brain-AI: Multimodal Decision Intelligence Engine

<div align="center">
  <img src="https://raw.githubusercontent.com/chandraveshchaudhari/chandraveshchaudhari/refs/heads/initial_setup/data/logo.png">
</div>


- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Contribution](#contribution)
- [Future Improvements](#future-improvements)

## Introduction
Brain-AI is a Multimodal Decision Intelligence Engine.

It is intentionally not a model-training framework by itself. Instead, it decides how multimodal pipelines are constructed, then executes those decisions through modular runtime components.

Pipeline decisions are represented as:

`PipelineSpec = (modalities, granularity_strategy, fusion_strategy, model_backend, hyperparameters)`

🚀 What it does
---------------
- Generates valid pipeline combinations through a dedicated decision layer.
- Executes multimodal pipelines through the `Brain` facade.
- Supports interchangeable granularity, fusion, and model adapter strategies.
- Produces DAG visualizations and structured experiment results.

🔥 Why it’s different
---------------------
- Explicit pipeline decision intelligence, separate from model backends.
- Modular architecture with single responsibility per module.
- User overrides always take precedence over automatic component resolution.


🧩 Features
-----------
- Granularity strategies: resample, pooling, attention-style alignment.
- Fusion strategies: early, late, intermediate.
- Thin model adapters: sklearn, AutoGluon scaffold, TPOT scaffold.
- Decision engine for generating and resolving `PipelineSpec` combinations.
- Experiment runner and leaderboard utilities.
- DAG generation for pipeline visualization.
- LLM skills for controlled external orchestration.

🧠 LLM Skills
------------
Skills expose safe, structured entry points for LLMs to generate specs,
run experiments, compare models and visualize DAGs via `brain_ai/agents/skills`.

🏗 Architecture Overview
------------------------
The codebase follows SRP and explicit boundaries:

- `brain_ai/core`: orchestrator (`Brain`) and compatibility exports.
- `brain_ai/decision`: `PipelineSpec`, combination generation, component resolution.
- `brain_ai/granularity`: alignment strategies.
- `brain_ai/fusion`: feature fusion strategies.
- `brain_ai/models`: thin backend adapters only.
- `brain_ai/experiments`: evaluator, runner, leaderboard.
- `brain_ai/dag`: DAG building and visualization.
- `brain_ai/rl`: policy search scaffolding (`state`, `action`, `reward`) with no algorithmic implementation.
- `brain_ai/agents`: planner/executor and LLM skills.
- `brain_ai/utils`: shared helpers (including synthetic multimodal dataset generator).

Design constraints enforced:

- No circular imports.
- No hidden coupling between granularity and fusion.
- Adapters remain thin wrappers.
- User override components always win over auto-resolved components.

For full developer docs, see the subpackage READMEs under `brain_ai/`.

## Installation 

⚡ Quick Start
-------------
Install the package (from source):

```bash
pip install -e .
```



### From PyPI
This project is available at [PyPI](https://pypi.org/project/brain-automl/). For help in installation check 
[instructions](https://packaging.python.org/tutorials/installing-packages/#installing-from-pypi)
```bash
python3 -m pip install brain-automl
```

Example usage:

```python
from brain_ai.decision.spec import PipelineSpec
from brain_ai.core.brain import Brain
from brain_ai.decision.engine import DecisionEngine

spec = PipelineSpec(modalities=["tabular"], granularity_strategy="resample",
                    fusion_strategy="early", model_backend="sklearn", hyperparameters={})

brain = Brain(decision_engine=DecisionEngine())
# run with multimodal data: {"modalities": {...}, "y": ...}
```

## Validation and Testing

Run the automated suite:

```bash
pytest -q
```

Covered checks include:

- different fusion strategies produce different outputs
- different granularity strategies produce different outputs
- full strategy combinations execute without error
- user overrides are respected
- DAG artifact generation works

Notebook walkthrough:

- `examples/multimodal_test.ipynb`


### Development Setup (Local Installation)

For development or running examples locally, follow these steps:

```zsh
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# OR
.\.venv\Scripts\Activate   # Windows

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Notebook Example

Use the end-to-end notebook walkthrough at:

- `examples/multimodal_test.ipynb`

The notebook includes:

- synthetic multimodal dataset generation
- pipeline combination generation
- batch pipeline execution and comparison table
- DAG generation for the best pipeline

If you are new to this repository, start with that notebook before exploring lower-level modules.

## Coverage

Run tests with coverage:

```bash
pytest --cov=brain_ai --cov-report=term-missing
```

## Important links
- [Documentation](https://chandraveshchaudhari.github.io/brain-ai/)
- [Quick tour](https://chandraveshchaudhari.github.io/brain-ai/brain-ai%20tutorial.html)
- [Project maintainer (feel free to contact)](mailto:chandraveshchaudhari@gmail.com?subject=[GitHub]%20Source%20brain-ai) 
- [Future Improvements](https://github.com/chandraveshchaudhari/brain-ai/projects)
- [License](https://github.com/chandraveshchaudhari/brain-ai/blob/master/LICENSE.txt)

## Contribution
all kinds of contributions are appreciated.
- [Improving readability of documentation](https://chandraveshchaudhari.github.io/brain-ai/)
- [Feature Request](https://github.com/chandraveshchaudhari/brain-ai/issues/new/choose)
- [Reporting bugs](https://github.com/chandraveshchaudhari/brain-ai/issues/new/choose)
- [Contribute code](https://github.com/chandraveshchaudhari/brain-ai/compare)
- [Asking questions in discussions](https://github.com/chandraveshchaudhari/brain-ai/discussions)
