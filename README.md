<div align="center">

# Graph-RAG-Engine
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange) ![Status](https://img.shields.io/badge/Status-Graph%20RAG%20MVP-green) [![CI](https://github.com/AmirhosseinHonardoust/Graph-RAG-Engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Graph-RAG-Engine/actions/workflows/ci.yml)

</div>

A professional **Graph + Vector Retrieval** project that indexes Markdown documents, builds a lightweight knowledge graph, retrieves relevant chunks with FAISS, expands results through graph concepts, and returns explainable answers with citations and reasoning paths.

> **Important:** This project is an **LLM-ready retrieval system**, not a fully autonomous knowledge engine.
>
> By default, it uses **extractive answer mode**, meaning it answers from retrieved passages instead of generating unsupported claims. Optional LLM mode can be enabled with an API key for grounded answer generation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Features](#features)
- [System Behavior](#system-behavior)
- [Architecture](#architecture)
- [Answer Modes](#answer-modes)
- [Retrieval Evaluation](#retrieval-evaluation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Building the Index](#building-the-index)
- [Running the Backend](#running-the-backend)
- [Running the Streamlit App](#running-the-streamlit-app)
- [API Usage](#api-usage)
- [Testing](#testing)
- [CI](#ci)
- [Limitations](#limitations)
- [Security Notes](#security-notes)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Many RAG projects behave like black boxes: a question goes in, an answer comes out, and the user cannot easily see why the system selected certain sources.

**Graph-RAG-Engine** takes a more explainable approach. It combines dense vector search with a lightweight graph layer so retrieved answers can include supporting citations, related concepts, and reasoning paths through documents, chunks, and concepts.

The goal of this project is to demonstrate:

- Hybrid retrieval with vector similarity and graph expansion
- Explainable source paths through a document/chunk/concept graph
- Extractive answers by default for safer local use
- Optional LLM-generated answers when credentials are provided
- Retrieval evaluation using golden queries and ranking metrics
- A testable project structure with CI support
- FastAPI and Streamlit interfaces for practical interaction

---

## What This Project Does

This project can:

- Read Markdown documents from `data/docs/`
- Split documents into text chunks
- Extract lightweight concepts from each chunk
- Create sentence embeddings with Sentence Transformers
- Build a FAISS vector index
- Build a NetworkX graph connecting documents, chunks, and concepts
- Retrieve relevant chunks for a user question
- Expand retrieval using graph concepts and document relationships
- Rerank results using similarity, concept overlap, and PageRank signals
- Return extractive answers with citations
- Return graph-based reasoning paths
- Recommend related documents
- Run retrieval evaluation with golden queries
- Run unit tests and CI checks
- Optionally generate grounded LLM answers using retrieved context

---

## What This Project Does Not Do

This project does **not**:

- Guarantee that every generated or retrieved answer is correct
- Replace human review for important research or business decisions
- Verify facts against the live web
- Use a production graph database by default
- Provide large-scale production observability or monitoring
- Make LLM access mandatory
- Claim that graph expansion always improves retrieval quality
- Protect against unsafe index artifacts from untrusted sources

A production-grade RAG platform would need stronger retrieval evaluation, source governance, monitoring, user feedback loops, access control, observability, and security hardening.

---

## Features

- **FAISS vector search** over normalized sentence embeddings
- **SentenceTransformer embeddings** using `all-MiniLM-L6-v2`
- **NetworkX knowledge graph** connecting documents, chunks, and concepts
- **Graph expansion** through shared concepts
- **Hybrid reranking** using similarity, concept overlap, and PageRank
- **Extractive answer mode** as the default behavior
- **Optional LLM answer mode** through environment variables
- **Citations and source paths** for explainability
- **Document recommendation endpoint**
- **Golden-query retrieval benchmark**
- **Retrieval metrics**: hit@k, precision@k, recall@k, MRR
- **FastAPI backend**
- **Streamlit frontend**
- **Unit tests**
- **GitHub Actions CI**

---

## System Behavior

<div align="center">
  
| Component | Current Behavior |
|---|---|
| Document source | Markdown files in `data/docs/` |
| Chunking | Simple paragraph-based chunking |
| Concept extraction | Lightweight keyword/frequency-based concept extraction |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS inner-product search over normalized embeddings |
| Graph layer | NetworkX graph connecting docs, chunks, and concepts |
| Retrieval expansion | Shared-concept expansion with configurable hop depth |
| Reranking | Embedding similarity + concept overlap + document PageRank |
| Default answer mode | Extractive answer composed from retrieved passages |
| Optional answer mode | LLM-generated answer using retrieved passages as source context |
| Evaluation | Golden-query retrieval benchmark with ranking metrics |
| Interface | FastAPI backend and Streamlit frontend |

</div>

---

## Architecture

```text
Markdown documents
        │
        ▼
Chunking + concept extraction
        │
        ├───────────────► Sentence embeddings ─────► FAISS vector index
        │
        └───────────────► Docs / chunks / concepts ─► NetworkX graph
                                                    │
                                                    ▼
Question ─► Vector search ─► Graph expansion ─► Hybrid reranking
                                                    │
                                                    ▼
                                   Extractive answer or optional LLM answer
                                                    │
                                                    ▼
                                  Citations + graph explanation paths
```

### Main Components

<div align="center">

| Component | Purpose |
|---|---|
| `ingest/` | Builds chunks, concepts, embeddings, FAISS index, and graph artifacts |
| `backend/retriever.py` | Loads retrieval artifacts lazily and performs hybrid retrieval |
| `backend/rag.py` | Orchestrates answer generation from retrieved sources |
| `backend/llm.py` | Provides optional OpenAI-compatible LLM answer generation |
| `graph/graph_store.py` | Stores and explains graph relationships |
| `evaluation/` | Runs retrieval benchmark metrics |
| `ui/app.py` | Provides a Streamlit interface |
</div>

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

If LLM mode is requested without a valid API key, the system falls back to extractive mode and returns an `llm_error` field.

---

## Retrieval Evaluation

The project includes a small retrieval benchmark in:

```text
evaluation/golden_queries.json
```

Run evaluation with:

```bash
python -m evaluation.evaluate_retrieval
```

The evaluation generates a report at:

```text
evaluation/results/retrieval_eval.json
```

### Evaluation Metrics

<div align="center">

| Metric | Meaning |
|---|---|
| `hit@k` | Whether at least one relevant document appears in the top-k results |
| `precision@k` | Fraction of top-k retrieved documents that are relevant |
| `recall@k` | Fraction of relevant documents retrieved in the top-k results |
| `MRR` | Reciprocal rank of the first relevant retrieved document |
</div>

Example custom run:

```bash
python -m evaluation.evaluate_retrieval \
  --k 3 \
  --base-k 8 \
  --top-n 6 \
  --expand-hops 1
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
│   ├── api.py
│   ├── llm.py
│   ├── rag.py
│   └── retriever.py
│
├── data/
│   ├── docs/
│   └── index/
│
├── env/
│   └── requirements.txt
│
├── evaluation/
│   ├── __init__.py
│   ├── golden_queries.json
│   ├── metrics.py
│   └── evaluate_retrieval.py
│
├── graph/
│   └── graph_store.py
│
├── ingest/
│   ├── ingest_docs.py
│   └── split.py
│
├── tests/
│   ├── test_graph_store.py
│   ├── test_llm.py
│   ├── test_project_integrity.py
│   ├── test_retrieval_evaluation_dataset.py
│   ├── test_retrieval_metrics.py
│   ├── test_retriever_api_contract.py
│   └── test_split.py
│
├── ui/
│   └── app.py
│
├── run_backend.py
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

### 3. Install Requirements

```bash
python -m pip install --upgrade pip
pip install -r env/requirements.txt
```

The first run may download the SentenceTransformer model, so an internet connection is required for initial setup.

---

## Building the Index

Run ingestion from the project root:

```bash
python -m ingest.ingest_docs
```

This will:

- read Markdown files from `data/docs/`
- split each document into chunks
- extract lightweight concepts
- create sentence embeddings
- save a FAISS index
- build and save the graph
- write retrieval artifacts to `data/index/`

Generated index artifacts are saved in:

```text
data/index/
```

---

## Running the Backend

Start the FastAPI backend:

```bash
uvicorn backend.api:app --reload --port 8000
```

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

The UI will open at:

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

---

## API Usage

### Ask a Question: Extractive Mode

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
      "url": "file:///.../data/docs/faiss_notes.md"
    }
  ],
  "paths": [
    {
      "chunk_id": "faiss_notes_chunk_0",
      "doc_id": "faiss_notes",
      "doc_title": "faiss_notes.md",
      "concepts": ["faiss", "vector", "search"]
    }
  ]
}
```

### Ask a Question: Optional LLM Mode

Set an API key:

```bash
export GRAPH_RAG_LLM_API_KEY="your_api_key_here"
```

Or use:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Optional configuration:

```bash
export GRAPH_RAG_LLM_MODEL="gpt-4o-mini"
export GRAPH_RAG_LLM_BASE_URL="https://api.openai.com/v1"
export GRAPH_RAG_LLM_TEMPERATURE="0.2"
export GRAPH_RAG_LLM_MAX_TOKENS="500"
export GRAPH_RAG_LLM_TIMEOUT_SECONDS="30"
```

Then call:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FAISS relate to embeddings?", "mode": "llm"}'
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

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The tests check important project behavior, including:

- Python source compilation
- Chunking behavior
- Concept extraction
- Graph neighbor lookup
- Graph save/load behavior
- Saved index consistency
- API health endpoint contract
- Retrieval metric calculations
- Golden-query dataset validation
- Optional LLM prompt/config behavior

---

## CI

The project includes a GitHub Actions workflow at:

```text
.github/workflows/ci.yml
```

CI runs on:

- push
- pull request
- manual dispatch

The workflow:

- installs dependencies from `env/requirements.txt`
- checks that source files compile
- runs the unit test suite
- tests against Python `3.10`, `3.11`, and `3.12`

---

## Limitations

This project has important limitations.

The system:

- uses a small sample corpus
- uses simple keyword/frequency-based concept extraction
- stores the graph in memory with NetworkX
- does not use a production graph database by default
- uses extractive answers by default
- depends on an external API provider for optional LLM mode
- uses a small golden-query retrieval benchmark
- does not include full production monitoring or observability
- may retrieve weak or incomplete context for ambiguous questions
- should not be treated as a fully reliable knowledge system

The project is best understood as a clean, testable, and explainable **Graph-RAG MVP**.

---

## Security Notes

- Index artifacts are loaded from local files, including pickle files.
- Only load artifacts generated by this project from trusted sources.
- Pickle files from untrusted sources can be unsafe.
- The FastAPI CORS configuration is permissive for local development.
- Restrict allowed origins before deploying publicly.
- Do not commit API keys.
- Use environment variables for LLM credentials.

---

## Responsible Use

This project is intended for:

- RAG architecture practice
- retrieval evaluation practice
- graph-based retrieval experimentation
- AI engineering portfolio demonstration
- FastAPI and Streamlit application development
- explainable retrieval interface design

It should not be used for:

- high-stakes decision-making
- legal, medical, financial, or safety-critical advice
- fully automated research conclusions
- private document processing without additional security controls
- production deployment without monitoring, access control, and audit logging

---

## Future Improvements

Possible future improvements include:

- Add a larger sample corpus
- Improve concept extraction with KeyBERT, spaCy noun chunks, YAKE, or embedding-based clustering
- Add more retrieval evaluation questions
- Add answer-quality evaluation for optional LLM mode
- Add screenshot or demo GIF of the UI
- Add Docker support
- Add a Neo4j-backed graph store option
- Add user feedback collection
- Add feedback-based reranking
- Add source-grounded answer faithfulness checks
- Add observability and retrieval trace logging

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
- Pickle artifacts
- unittest
- GitHub Actions

---

## Author

**Amir Honardoust**  
GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is released under the MIT License.
