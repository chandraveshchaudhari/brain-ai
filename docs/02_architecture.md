# Brain-AI Architecture

## High-Level Architecture

User

↓

UI Layer

↓

LLM Agent

↓

Tool Registry

↓

AutoML Libraries

↓

Results

---

## Architecture Layers

### 1. UI Layer

* Streamlit or web based ui
* VS Code UI
* CLI

---

### 2. LLM Agent Layer

Responsibilities:

* Dataset understanding
* Pipeline generation
* Tool selection
* Planning

---

### 3. Tool Layer

Tools include:

* Dataset analyzer
* Preprocessing
* Fusion
* AutoML execution
* Evaluation

---

## Tool Structure

brain-ai/

tools/

* analyze_dataset.py
* preprocess.py
* fusion.py
* run_automl.py

---

## Tool Registry

registry.py

Example:

TOOLS = [
analyze_dataset,
preprocess,
run_automl
]

---

## Agent Flow

User uploads dataset

↓

LLM planner

↓

Select tools

↓

Execute pipeline

↓

Return results

---

## CLI Architecture

brain-ai analyze dataset.csv

brain-ai run

brain-ai evaluate

---

## UI Flow

Upload dataset

↓

Analyze

↓

Generate pipeline

↓

Run models

↓

Show results

---

## Future Architecture

Add:

* Reinforcement learning
* Self improving pipeline
* Multi agent system
