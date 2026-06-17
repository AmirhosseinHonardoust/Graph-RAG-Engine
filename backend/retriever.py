from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_store import GraphStore

BASE = Path(__file__).resolve().parents[1]
IDX_DIR = BASE / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RANKING_WEIGHTS: dict[str, float] = {
    "embedding_similarity": 0.60,
    "concept_overlap": 0.25,
    "doc_pagerank": 0.15,
}


@dataclass(frozen=True)
class RetrieverStore:
    """In-memory retrieval artifacts.

    The original implementation loaded the FAISS index, graph, chunks, vectors,
    and embedding model at import time. This made simple imports expensive and
    harder to test. The store is now loaded lazily through ``get_store``.
    """

    chunks: list[dict[str, Any]]
    vecs: np.ndarray
    index: Any
    model: SentenceTransformer
    graph: GraphStore
    chunk_by_id: dict[str, dict[str, Any]]
    chunk_index_by_id: dict[str, int]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing retrieval artifact: {path}. "
            "Run `python -m ingest.ingest_docs` before starting the API."
        )


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_store(index_dir: str | Path = IDX_DIR, model_name: str = MODEL_NAME) -> RetrieverStore:
    """Load and cache retrieval artifacts.

    Parameters are kept explicit so tests and future deployments can point to a
    different index directory or model name without editing module globals.
    """

    index_dir = Path(index_dir)
    required = [
        index_dir / "chunks.json",
        index_dir / "vectors.npy",
        index_dir / "faiss.index",
        index_dir / "graph.json",
    ]
    for path in required:
        _require_file(path)

    chunks = _load_json(index_dir / "chunks.json")
    vecs = np.load(index_dir / "vectors.npy").astype(np.float32)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    graph = GraphStore.load(index_dir / "graph.json")
    model = SentenceTransformer(model_name)

    chunk_by_id = {c["id"]: c for c in chunks}
    chunk_index_by_id = {c["id"]: i for i, c in enumerate(chunks)}

    if len(chunks) != len(vecs):
        raise ValueError(f"Index artifact mismatch: {len(chunks)} chunks but {len(vecs)} vectors.")

    return RetrieverStore(
        chunks=chunks,
        vecs=vecs,
        index=index,
        model=model,
        graph=graph,
        chunk_by_id=chunk_by_id,
        chunk_index_by_id=chunk_index_by_id,
    )


def ann_search(q: str, k: int = 8, store: RetrieverStore | None = None) -> list[tuple[str, float]]:
    """Run approximate nearest-neighbor search over chunk embeddings."""

    if not q or not q.strip():
        return []

    store = store or get_store()
    k = max(1, min(k, len(store.chunks)))

    qv = store.model.encode([q], normalize_embeddings=True).astype(np.float32)
    distances, indices = store.index.search(qv, k)

    results: list[tuple[str, float]] = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        results.append((store.chunks[int(idx)]["id"], float(distances[0][rank])))
    return results


def _query_terms(q: str) -> set[str]:
    """Extract normalized query terms used for concept overlap."""

    return {term.lower() for term in re.findall(r"[A-Za-z0-9_+-]+", q) if len(term) > 2}


def _normalize_concepts(concepts: Iterable[Any]) -> set[str]:
    return {str(concept).strip().lower() for concept in concepts if str(concept).strip()}


def concept_overlap_score(
    q_terms: set[str], concepts: Iterable[Any]
) -> tuple[float, int, list[str]]:
    """Return normalized concept-overlap score and matching concepts.

    The previous reranker used the raw overlap count. Normalizing by the number
    of query terms keeps the concept-overlap component in a comparable 0-1 range
    with embedding similarity and PageRank-style scores.
    """

    normalized_concepts = _normalize_concepts(concepts)
    if not q_terms:
        return 0.0, 0, []

    matches = sorted(q_terms.intersection(normalized_concepts))
    score = len(matches) / max(1, len(q_terms))
    return float(score), len(matches), matches


def build_retrieval_trace(
    *,
    question_vector: np.ndarray,
    query_terms: set[str],
    chunk: dict[str, Any],
    chunk_index: int,
    store: RetrieverStore,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build transparent score components for a candidate chunk."""

    weights = dict(weights or RANKING_WEIGHTS)
    embedding_similarity = float(np.dot(question_vector, store.vecs[chunk_index]))
    concept_overlap, concept_overlap_count, matched_concepts = concept_overlap_score(
        query_terms,
        chunk.get("concepts", []),
    )
    doc_pagerank = float(store.graph.get_doc_info(chunk["doc_id"]).get("pagerank", 0.0))
    final_score = (
        weights["embedding_similarity"] * embedding_similarity
        + weights["concept_overlap"] * concept_overlap
        + weights["doc_pagerank"] * doc_pagerank
    )

    return {
        "chunk_id": chunk["id"],
        "doc_id": chunk.get("doc_id"),
        "doc_title": chunk.get("doc_title"),
        "embedding_similarity": embedding_similarity,
        "concept_overlap": concept_overlap,
        "concept_overlap_count": concept_overlap_count,
        "matched_concepts": matched_concepts,
        "doc_pagerank": doc_pagerank,
        "weights": weights,
        "final_score": float(final_score),
    }


def _expand_chunk_ids_by_concepts(
    chunk_ids: Iterable[str],
    store: RetrieverStore,
    hops: int,
    max_neighbors: int = 8,
) -> set[str]:
    """Expand a chunk set by walking shared-concept chunk links."""

    visited = set(chunk_ids)
    frontier = set(chunk_ids)

    for _ in range(max(0, hops)):
        next_frontier: set[str] = set()
        for cid in frontier:
            for neighbor in store.graph.neighbor_chunks_by_concepts(
                cid, max_neighbors=max_neighbors
            ):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return visited


def expand_and_rerank(
    q: str,
    base_k: int = 8,
    expand_hops: int = 1,
    top_n: int = 6,
    store: RetrieverStore | None = None,
) -> list[dict[str, Any]]:
    """Retrieve, graph-expand, and rerank candidate chunks.

    Score blends embedding similarity, normalized query/concept overlap, and
    document PageRank. Each returned passage includes a ``retrieval_trace`` so
    API clients can inspect why a chunk was ranked.
    """

    store = store or get_store()
    base = ann_search(q, k=base_k, store=store)
    if not base:
        return []

    base_ids = [cid for cid, _ in base]
    candidate_ids = _expand_chunk_ids_by_concepts(
        base_ids,
        store=store,
        hops=expand_hops,
        max_neighbors=6,
    )

    qv = store.model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
    q_terms = _query_terms(q)

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for cid in candidate_ids:
        chunk = store.chunk_by_id[cid]
        idx = store.chunk_index_by_id[cid]
        trace = build_retrieval_trace(
            question_vector=qv,
            query_terms=q_terms,
            chunk=chunk,
            chunk_index=idx,
            store=store,
        )
        scored.append((trace["final_score"], cid, trace))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    passages: list[dict[str, Any]] = []
    for _, cid, trace in scored[: max(1, top_n)]:
        passage = dict(store.chunk_by_id[cid])
        passage["retrieval_trace"] = trace
        passages.append(passage)
    return passages


def recommend_similar(
    doc_id: str,
    k: int = 5,
    store: RetrieverStore | None = None,
) -> list[dict[str, Any]]:
    """Recommend documents whose chunks are close to the selected document."""

    store = store or get_store()
    chunk_indices = [i for i, chunk in enumerate(store.chunks) if chunk.get("doc_id") == doc_id]
    if not chunk_indices:
        return []

    centroid = store.vecs[chunk_indices].mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    search_k = max(1, min(32, len(store.chunks)))
    _, indices = store.index.search(centroid.reshape(1, -1), search_k)

    doc_scores: dict[str, float] = {}
    for idx in indices[0]:
        if idx < 0:
            continue
        candidate_doc_id = store.chunks[int(idx)]["doc_id"]
        if candidate_doc_id == doc_id:
            continue
        sim = float(np.dot(centroid, store.vecs[int(idx)]))
        doc_scores[candidate_doc_id] = max(doc_scores.get(candidate_doc_id, 0.0), sim)

    for candidate_doc_id in list(doc_scores.keys()):
        doc_scores[candidate_doc_id] = 0.8 * doc_scores[
            candidate_doc_id
        ] + 0.2 * store.graph.get_doc_info(candidate_doc_id).get("pagerank", 0.0)

    recommendations = [
        {"doc_id": candidate_doc_id, **store.graph.get_doc_info(candidate_doc_id), "score": score}
        for candidate_doc_id, score in doc_scores.items()
    ]
    return sorted(recommendations, key=lambda x: -x["score"])[: max(1, k)]
