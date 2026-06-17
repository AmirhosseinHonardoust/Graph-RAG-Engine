from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_store import GraphStore

from .split import extract_concepts, simple_chunk

BASE = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE / "data" / "docs"
OUT_DIR = BASE / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def repo_relative_path(path: Path) -> str:
    """Return a stable POSIX-style path relative to the repository root."""

    return path.resolve().relative_to(BASE).as_posix()


def load_docs(docs_dir: Path = DOCS_DIR) -> list[dict[str, Any]]:
    """Load markdown documents in a deterministic order.

    The previous implementation stored absolute local ``file://`` URLs inside
    the index artifacts. That leaked machine-specific paths such as
    ``C:\\Users\\...`` and made citations non-portable. This loader stores
    repository-relative paths instead, for example ``data/docs/faiss_notes.md``.
    """

    docs: list[dict[str, Any]] = []
    docs_dir = Path(docs_dir)
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            {
                "id": path.stem,
                "title": path.name,
                "url": repo_relative_path(path),
                "text": text,
            }
        )
    return docs


def build_index(docs: list[dict[str, Any]], out_dir: Path = OUT_DIR) -> None:
    """Build and persist chunk, vector, FAISS, and graph artifacts."""

    if not docs:
        raise ValueError("Cannot build an index without documents.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    for doc in docs:
        for i, chunk_text in enumerate(simple_chunk(doc["text"])):
            chunk_id = f"{doc['id']}_chunk_{i}"
            chunks.append(
                {
                    "id": chunk_id,
                    "doc_id": doc["id"],
                    "doc_title": doc["title"],
                    "url": doc["url"],
                    "text": chunk_text,
                    "concepts": extract_concepts(chunk_text),
                }
            )

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True)
    vectors = vectors.astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    with (out_dir / "docs.pkl").open("wb") as f:
        pickle.dump(docs, f)
    with (out_dir / "chunks.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    np.save(out_dir / "vectors.npy", vectors)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    graph_store = GraphStore()
    for doc in docs:
        graph_store.add_doc(doc["id"], doc["title"], doc["url"])
    for chunk in chunks:
        graph_store.add_chunk(chunk["id"], chunk["text"], chunk["doc_id"])
        for concept in chunk["concepts"]:
            graph_store.add_concept(concept)
            graph_store.link_mentions(chunk["id"], concept)
    graph_store.compute_doc_pagerank()
    graph_store.save(out_dir / "graph.pkl")


if __name__ == "__main__":
    loaded_docs = load_docs()
    if not loaded_docs:
        raise SystemExit(f"No docs found in {DOCS_DIR}")
    build_index(loaded_docs)
    print(f"Ingested {len(loaded_docs)} docs. Index written to {OUT_DIR}")
