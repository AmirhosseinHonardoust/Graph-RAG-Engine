<div align="center">
  
# Graph-RAG-Engine
<img width="1666" height="935" alt="Graph-RAG-Engine" src="https://github.com/user-attachments/assets/080172f2-0a85-4c97-8f3f-2e83b9272720" />

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Knowledge%20Graph-purple)
![Status](https://img.shields.io/badge/Status-Portfolio%20MVP-green)
[![CI](https://github.com/AmirhosseinHonardoust/Graph-RAG-Engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Graph-RAG-Engine/actions/workflows/ci.yml)

</div>

An explainable **Graph + Vector Retrieval** system that indexes Markdown documents, builds a lightweight knowledge graph, retrieves relevant chunks with FAISS, expands results through graph concepts, and returns source-grounded answers with citations, reasoning paths, and retrieval traces.

> **Important:** This project is an **LLM-ready retrieval system**, not a fully autonomous knowledge engine.
>
> By default, it uses **extractive answer mode**, meaning it answers from retrieved passages instead of generating unsupported claims. Optional LLM mode can be enabled with an API key for grounded answer generation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Building the Index](#building-the-index)
- [Running the Backend](#running-the-backend)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Answer Modes](#answer-modes)
- [API Usage](#api-usage)
- [Retrieval Trace](#retrieval-trace)
- [Retrieval Evaluation](#retrieval-evaluation)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Security Notes](#security-notes)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Many RAG projects behave like black boxes: a question goes in, an answer comes out, and the user cannot easily see why the system selected certain sources.

**Graph-RAG-Engine** takes a more transparent approach. It combines dense vector search with a lightweight graph layer so answers can include retrieved passages, citations, related concepts, graph paths, and score-level retrieval traces.

The goal is to demonstrate how a RAG workflow can be turned into an explainable retrieval system, not just a chatbot wrapper around an embedding index.

This project focuses on:

- document ingestion from local Markdown files
- chunking and lightweight concept extraction
- dense vector search with FAISS
- graph-enhanced retrieval with NetworkX
- hybrid reranking using similarity, concept overlap, and PageRank
- extractive answers by default
- optional LLM-generated answers when credentials are provided
- golden-query retrieval evaluation
- FastAPI and Streamlit interfaces
- tests and CI for safer iteration

---

## What This Project Does

This project can:

- Read Markdown documents from `data/docs/`
- Split documents into retrievable text chunks
- Extract lightweight concepts from each chunk
- Create sentence embeddings with Sentence Transformers
- Build a FAISS vector index
- Build a NetworkX graph connecting documents, chunks, and concepts
- Store repository-relative source paths inside index artifacts
- Retrieve relevant chunks for a user question
- Expand retrieval through graph concepts and document relationships
- Rerank results using embedding similarity, normalized concept overlap, and document PageRank
- Return extractive answers with citations
- Return graph-based reasoning paths
- Return retrieval traces with score components
- Recommend related documents
- Run retrieval evaluation with golden queries
- Run unit tests and CI checks
- Optionally generate grounded LLM answers from retrieved context
- Use Streamlit to ask questions in extractive or optional LLM mode

---

## What This Project Does Not Do

This project does **not**:

- Guarantee that every retrieved or generated answer is correct
- Replace human review for important research or business decisions
- Verify facts against the live web
- Use a production graph database by default
- Provide large-scale observability or monitoring
- Make LLM access mandatory
- Claim that graph expansion always improves retrieval quality
- Protect against unsafe index artifacts from untrusted sources
- Provide production-grade access control, audit logging, or permissioning

A production RAG platform would need stronger retrieval evaluation, source governance, monitoring, feedback loops, access control, observability, data-security review, and deployment hardening.

---

## Key Features

- **FAISS vector search** over normalized sentence embeddings
- **SentenceTransformer embeddings** using `all-MiniLM-L6-v2`
- **NetworkX knowledge graph** connecting documents, chunks, and concepts
- **Graph expansion** through shared concepts
- **Hybrid reranking** using vector similarity, concept overlap, and PageRank
- **Normalized concept-overlap scoring** for more balanced reranking
- **Retrieval traces** showing score components and matched concepts
- **Extractive answer mode** as the default behavior
- **Optional LLM answer mode** with fallback to extractive mode
- **Streamlit answer-mode selector** for extractive or LLM mode
- **Repository-relative index paths** instead of local machine paths
- **Expanded demo corpus** with multiple RAG engineering topics
- **Golden-query retrieval benchmark**
- **Retrieval metrics**: Hit@K, Precision@K, Recall@K, MRR, and nDCG@K
- **FastAPI backend**
- **Streamlit frontend**
- **Unit tests and GitHub Actions CI**
- **CI retrieval evaluation smoke workflow**

---

## System Workflow

```text
Markdown documents
        ↓
Chunking + concept extraction
        ↓
Sentence embeddings ───────────────→ FAISS vector index
        ↓
Docs / chunks / concepts ──────────→ NetworkX graph
        ↓
User question
        ↓
Vector search
        ↓
Graph expansion
        ↓
Hybrid reranking
        ↓
Extractive answer or optional LLM answer
        ↓
Citations + graph paths + retrieval traces
```

---

## Project Structure

```text
Graph-RAG-Engine/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── llm.py
│   ├── rag.py
│   └── retriever.py
│
├── data/
│   ├── docs/
│   │   ├── api_contracts.md
│   │   ├── chunking_strategies.md
│   │   ├── citation_grounding.md
│   │   ├── graph_rag_overview.md
│   │   ├── hybrid_retrieval.md
│   │   ├── retrieval_evaluation.md
│   │   └── ...
│   └── index/                  # generated by ingestion (gitignored)
│       ├── chunks.json
│       ├── docs.json
│       ├── faiss.index
│       ├── graph.json
│       └── vectors.npy
│
├── env/
│   └── requirements.txt
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_retrieval.py
│   ├── baseline.json
│   ├── golden_queries.json
│   ├── metrics.py
│   └── results/              # generated by evaluation (gitignored)
│       └── retrieval_eval.json
│
├── graph/
│   ├── __init__.py
│   └── graph_store.py
│
├── ingest/
│   ├── __init__.py
│   ├── ingest_docs.py
│   └── split.py
│
├── tests/
│   ├── test_evaluation_baseline.py
│   ├── test_graph_store.py
│   ├── test_llm.py
│   ├── test_project_integrity.py
│   ├── test_retrieval_evaluation_dataset.py
│   ├── test_retrieval_evaluation_smoke.py
│   ├── test_retrieval_metrics.py
│   ├── test_retrieval_trace.py
│   ├── test_retriever_api_contract.py
│   ├── test_split.py
│   └── test_ui_contracts.py
│
├── ui/
│   ├── __init__.py
│   └── app.py
│
├── pipeline.py
├── run_backend.py
├── Makefile
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Graph-RAG-Engine.git
cd Graph-RAG-Engine
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the Package

Install in editable mode. This pulls in all dependencies and registers the
`graph-rag-ingest`, `graph-rag-eval`, and `graph-rag-serve` commands:

```bash
python -m pip install --upgrade pip
pip install -e .
```

For development tools (Ruff, Black, mypy, pre-commit):

```bash
pip install -e ".[dev]"
```

The pinned dependency list is also available at `env/requirements.txt`. The
first embedding run may download the SentenceTransformer model, so an internet
connection is required for initial setup.

---

## Quick Start

Once installed, build the index and validate retrieval quality in one command:

```bash
graph-rag-pipeline
```

This ingests the documents, builds the index, runs the golden-query evaluation,
and fails if any metric drops below `evaluation/baseline.json`. Common tasks are
also wrapped in a `Makefile`:

```bash
make dev       # install with dev tools and git hooks
make pipeline  # ingest + evaluate
make serve     # run the FastAPI backend
make lint      # ruff + black + mypy
make test      # run the unit tests
```

On platforms without `make`, use the console commands directly
(`graph-rag-ingest`, `graph-rag-eval`, `graph-rag-serve`, `graph-rag-pipeline`).

---

## Building the Index

Run ingestion (from anywhere once installed):

```bash
graph-rag-ingest
```

This will:

- read Markdown files from `data/docs/`
- sort document loading deterministically
- split each document into chunks
- extract lightweight concepts
- create sentence embeddings
- save a FAISS index
- build and save the graph
- store repo-relative source paths
- write retrieval artifacts to `data/index/`

Generated index artifacts are saved in:

```text
data/index/
```

> **Security note:** Index artifacts (`chunks.json`, `docs.json`, `graph.json`) are stored as JSON, so loading them cannot execute code. The FAISS index and `vectors.npy` are binary numeric files. Only the embedding vectors and FAISS index require trusting the producer; none of the artifacts are pickled.

---

## Running the Backend

Start the FastAPI backend:

```bash
graph-rag-serve
```

This serves `backend.api:app` on `127.0.0.1:8000` with reload enabled. You can
also run uvicorn directly: `uvicorn backend.api:app --reload --port 8000`.

Open the API docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "ok": true
}
```

---

## Running the Streamlit App

In another terminal, run:

```bash
streamlit run ui/app.py
```

The UI opens at:

```text
http://localhost:8501
```

By default, the UI calls the backend at:

```text
http://localhost:8000
```

You can override this with:

```bash
export GRAPH_RAG_API_URL="http://localhost:8000"
```

On Windows PowerShell:

```powershell
$env:GRAPH_RAG_API_URL="http://localhost:8000"
```

The Streamlit app supports both answer modes:

- `extractive`
- `llm`

If LLM mode is selected without a valid API key, the backend falls back to extractive mode and returns an LLM error message.

---

## Answer Modes

The project supports two answer modes.

### 1. Extractive Mode

Extractive mode is the default.

It uses retrieved passages directly and avoids generating unsupported claims.

```json
{
  "question": "What is FAISS?",
  "mode": "extractive"
}
```

This mode is useful for:

- local demos
- transparent retrieval debugging
- source-first answer display
- running the system without API keys

### 2. Optional LLM Mode

LLM mode sends retrieved passages to an OpenAI-compatible chat-completions API and asks the model to answer only from the provided context.

```json
{
  "question": "How does FAISS relate to embeddings?",
  "mode": "llm"
}
```

Configure optional LLM mode with environment variables:

```bash
export GRAPH_RAG_LLM_API_KEY="your_api_key_here"
export GRAPH_RAG_LLM_MODEL="gpt-4o-mini"
export GRAPH_RAG_LLM_BASE_URL="https://api.openai.com/v1"
export GRAPH_RAG_LLM_TEMPERATURE="0.2"
export GRAPH_RAG_LLM_MAX_TOKENS="500"
export GRAPH_RAG_LLM_TIMEOUT_SECONDS="30"
```

You can also use:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If LLM mode is requested without valid credentials, the system returns an extractive answer and includes an `llm_error` field.

---

## API Usage

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FAISS?", "mode": "extractive"}'
```

Example response shape:

```json
{
  "answer": "...",
  "answer_mode": "extractive",
  "citations": [
    {
      "doc_title": "faiss_notes.md",
      "url": "data/docs/faiss_notes.md"
    }
  ],
  "paths": [
    {
      "chunk_id": "faiss_notes_chunk_0",
      "doc_id": "faiss_notes",
      "doc_title": "faiss_notes.md",
      "concepts": ["faiss", "vector", "search"]
    }
  ],
  "retrieval_traces": [
    {
      "chunk_id": "faiss_notes_chunk_0",
      "embedding_similarity": 0.72,
      "concept_overlap": 0.33,
      "doc_pagerank": 0.04,
      "final_score": 0.52
    }
  ]
}
```

### Recommend Similar Documents

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "faiss_notes"}'
```

### List Indexed Documents

```bash
curl http://localhost:8000/docs_list
```

---

## Retrieval Trace

The retrieval trace makes reranking easier to inspect.

Each retrieved passage can include score components such as:

<div align="center">

| Field | Meaning |
|---|---|
| `chunk_id` | Retrieved chunk identifier |
| `doc_id` | Source document identifier |
| `doc_title` | Source document title |
| `embedding_similarity` | Dense vector similarity score |
| `concept_overlap` | Normalized query/document concept overlap |
| `concept_overlap_count` | Number of matched concepts |
| `matched_concepts` | Concepts shared with the query |
| `doc_pagerank` | Graph PageRank signal for the document |
| `weights` | Reranking weights used by the retriever |
| `final_score` | Final hybrid retrieval score |
</div>

The current reranking formula is:

```text
final_score =
  0.60 * embedding_similarity
+ 0.25 * normalized_concept_overlap
+ 0.15 * doc_pagerank
```

This does not prove the result is correct, but it makes the retrieval behavior easier to debug and explain.

---

## Retrieval Evaluation

The project includes a golden-query retrieval benchmark in:

```text
evaluation/golden_queries.json
```

The expanded demo benchmark includes:

<div align="center">

| Item | Count |
|---|---|
| Markdown documents | 19 |
| Golden retrieval queries | 35 |
| Multi-document query cases | Included |
| Retrieval topics | Graph-RAG, chunking, hybrid retrieval, citations, evaluation, API contracts, failure modes, and production considerations |
</div>

Run evaluation with:

```bash
graph-rag-eval
```

The evaluation generates:

```text
evaluation/results/retrieval_eval.json
```

Evaluation metrics include:

<div align="center">

| Metric | Meaning |
|---|---|
| `hit@k` | Whether at least one relevant document appears in the top-k results |
| `precision@k` | Fraction of top-k retrieved documents that are relevant |
| `recall@k` | Fraction of relevant documents retrieved in the top-k results |
| `MRR` | Reciprocal rank of the first relevant retrieved document |
| `nDCG@k` | Rank-weighted gain, rewarding relevant documents nearer the top |
</div>

Example custom run:

```bash
python -m evaluation.evaluate_retrieval \
  --k 3 \
  --base-k 8 \
  --top-n 6 \
  --expand-hops 1
```

### Regression gate

`evaluation/baseline.json` holds minimum acceptable metrics. Passing
`--fail-under evaluation/baseline.json` makes the evaluation exit non-zero if any
metric drops below its threshold, which CI uses to block retrieval regressions:

```bash
graph-rag-eval --fail-under evaluation/baseline.json
```

> The included benchmark is a demo retrieval benchmark, not a production-grade RAG evaluation suite.

---

## Testing and CI

Run the test suite locally:

```bash
python -m unittest discover -s tests -v
```

Compile source files:

```bash
python -m compileall backend graph ingest ui evaluation tests run_backend.py
```

The tests check important project behavior, including:

- Python source compilation
- chunking behavior
- concept extraction
- graph neighbor lookup
- graph save/load behavior
- saved index consistency
- local-path safety in index artifacts
- API health endpoint contract
- retrieval metric calculations
- golden-query dataset validation
- retrieval trace behavior
- optional LLM prompt/config behavior
- Streamlit UI request contracts

CI is defined in:

```text
.github/workflows/ci.yml
```

The GitHub Actions workflow checks:

- dependency installation
- source compilation
- unit tests
- index rebuild smoke workflow
- retrieval evaluation smoke workflow
- expected FAISS/vector/graph artifacts
- expected retrieval evaluation report
- artifact upload for generated retrieval outputs

---

## Code Quality

The project separates major responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `ingest/ingest_docs.py` | Builds document, chunk, embedding, FAISS, and graph artifacts |
| `ingest/split.py` | Splits Markdown text into retrievable chunks |
| `backend/retriever.py` | Loads retrieval artifacts lazily and performs hybrid retrieval |
| `backend/rag.py` | Orchestrates answer generation, citations, paths, and traces |
| `backend/llm.py` | Provides optional OpenAI-compatible LLM answer generation |
| `backend/api.py` | Exposes FastAPI endpoints |
| `graph/graph_store.py` | Stores and explains graph relationships |
| `evaluation/metrics.py` | Computes retrieval ranking metrics |
| `evaluation/evaluate_retrieval.py` | Runs golden-query retrieval evaluation |
| `ui/app.py` | Provides the Streamlit interface |
</div>

The current implementation is intentionally lightweight and local-first, which makes it suitable for a portfolio MVP and for experimenting with retrieval behavior.

---

## Limitations

This project has important limitations:

- The demo corpus is still small compared with real knowledge bases
- Concept extraction is lightweight and keyword/frequency based
- The graph is stored in memory with NetworkX
- It does not use a production graph database by default
- The retrieval benchmark is useful but not exhaustive
- Retrieval quality depends heavily on chunking and embedding quality
- Optional LLM mode depends on an external API provider
- The system may retrieve weak or incomplete context for ambiguous questions
- It does not include production observability or monitoring
- It does not include access control, user permissions, or document-level authorization
- It should not be treated as a fully reliable knowledge system

The project is strongest as a clean, testable, and explainable **Graph-RAG portfolio MVP**.

---

## Security Notes

- Index artifacts are loaded from local files. Chunk, document, and graph data are JSON (no code execution on load); the FAISS index and vectors are binary numeric files.
- Only load artifacts generated by this project from trusted sources.
- The FastAPI CORS configuration defaults to permissive (`*`) for local development.
- Set `GRAPH_RAG_ALLOW_ORIGINS` (comma-separated) to restrict origins before deploying publicly; credentials are automatically disabled while origins are wildcarded.
- Do not commit API keys.
- Use environment variables for LLM credentials.
- Do not index private documents without access-control and data-governance review.

---

## Responsible Use

This repository is intended for:

- RAG architecture practice
- retrieval evaluation practice
- graph-based retrieval experimentation
- AI engineering portfolio demonstration
- FastAPI and Streamlit application development
- explainable retrieval interface design

It should not be used as-is for:

- high-stakes decision-making
- legal, medical, financial, or safety-critical advice
- fully automated research conclusions
- private document processing without additional security controls
- production deployment without monitoring, access control, and audit logging

Any real deployment would require stronger source governance, monitoring, feedback review, security controls, evaluation, and human oversight.

---

## Future Improvements

Potential next improvements:

- Add a larger and more realistic sample corpus
- Add answer-quality and faithfulness evaluation for LLM mode
- Add screenshot or demo GIF of the Streamlit UI
- Add Docker support
- Add a Neo4j-backed graph store option
- Add KeyBERT, spaCy noun chunks, YAKE, or embedding-clustering concept extraction
- Add feedback-based reranking
- Add source-grounded answer faithfulness checks
- Add retrieval trace logging
- Add document-level permissions
- Add observability and monitoring examples
- Add benchmark comparison between vector-only and graph-expanded retrieval

---

## Tech Stack

- Python
- FastAPI
- Pydantic
- Uvicorn
- Streamlit
- Sentence Transformers
- FAISS
- NetworkX
- NumPy
- Markdown files
- JSON artifacts
- unittest
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is released under the MIT License.

If you use or modify this project, keep the limitations, security notes, and responsible-use guidance clear.
