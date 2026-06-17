from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import log2
from statistics import mean


@dataclass(frozen=True)
class QueryEvaluation:
    """Retrieval metrics for a single golden query."""

    query_id: str
    question: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    k: int
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "relevant_doc_ids": self.relevant_doc_ids,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "k": self.k,
            "hit_at_k": self.hit_at_k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
        }


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Return unique items without changing their first-seen order."""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _top_k(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return list(retrieved[:k])


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(any(doc_id in relevant for doc_id in _top_k(retrieved, k)))


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top = _top_k(retrieved, k)
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = _top_k(retrieved, k)
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at k with binary relevance."""
    if not relevant:
        return 0.0
    top = _top_k(retrieved, k)
    dcg = sum(
        1.0 / log2(rank + 1) for rank, doc_id in enumerate(top, start=1) if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    *,
    query_id: str,
    question: str,
    relevant_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    k: int,
) -> QueryEvaluation:
    """Evaluate a ranked document list against one golden query."""

    ranked_unique = unique_preserve_order(retrieved_doc_ids)
    relevant = set(relevant_doc_ids)
    return QueryEvaluation(
        query_id=query_id,
        question=question,
        relevant_doc_ids=list(relevant_doc_ids),
        retrieved_doc_ids=ranked_unique,
        k=k,
        hit_at_k=hit_at_k(ranked_unique, relevant, k),
        precision_at_k=precision_at_k(ranked_unique, relevant, k),
        recall_at_k=recall_at_k(ranked_unique, relevant, k),
        reciprocal_rank=reciprocal_rank(ranked_unique, relevant),
        ndcg_at_k=ndcg_at_k(ranked_unique, relevant, k),
    )


def summarize_evaluations(evaluations: Sequence[QueryEvaluation]) -> dict[str, float]:
    """Aggregate query-level evaluations into mean metrics."""

    if not evaluations:
        return {
            "num_queries": 0,
            "mean_hit_at_k": 0.0,
            "mean_precision_at_k": 0.0,
            "mean_recall_at_k": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_ndcg_at_k": 0.0,
        }

    return {
        "num_queries": float(len(evaluations)),
        "mean_hit_at_k": mean(e.hit_at_k for e in evaluations),
        "mean_precision_at_k": mean(e.precision_at_k for e in evaluations),
        "mean_recall_at_k": mean(e.recall_at_k for e in evaluations),
        "mean_reciprocal_rank": mean(e.reciprocal_rank for e in evaluations),
        "mean_ndcg_at_k": mean(e.ndcg_at_k for e in evaluations),
    }
