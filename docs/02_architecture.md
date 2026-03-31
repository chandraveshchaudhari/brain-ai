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

* Streamlit
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

# Current Architecture

## Proposed Pipeline

PDFs (100s)
↓
Docling parser
↓
Chunking (500 tokens)
↓
BGE embeddings
↓
Qdrant vector DB
↓
Hybrid search
↓
RAG system
↓
FastAPI backend
↓
CloudFront + Lambda deployment

---

## Components

### 1. PDF Loader

- Docling
- PyMuPDF
- PDFPlumber

Preferred: Docling (structure-aware parsing)

---

### 2. Chunking

Options:

- Fixed token chunking
- Semantic chunking
- Section-aware chunking

Recommended:

- Section-aware chunking

---

### 3. Embeddings

Proposed:

- BGE embeddings

Options:

- bge-large
- bge-m3
- e5-large
- instructor-xl

---

### 4. Vector Database

Chosen:

- Qdrant

Alternatives:

- Weaviate
- Pinecone
- Chroma

---

### 5. Hybrid Search

Combine:

- Dense embeddings
- Sparse retrieval (BM25)

Benefits:

- Better accuracy
- Better recall

---

### 6. RAG Layer

Options:

- LangChain
- LlamaIndex
- Custom RAG

Preferred:

- Custom RAG (more control)

---

### 7. Backend

- FastAPI

---

### 8. Deployment

- Lambda
- API Gateway
- CloudFront