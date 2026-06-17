from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .metrics import evaluate_query, summarize_evaluations

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "evaluation" / "golden_queries.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "retrieval_eval.json"


def load_golden_queries(path: str | Path = DEFAULT_QUERIES) -> list[dict[str, Any]]:
    """Load and validate the golden retrieval queries file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list) or not queries:
        raise ValueError("Golden query file must contain a non-empty list.")

    required = {"id", "question", "relevant_doc_ids"}
    for i, query in enumerate(queries):
        missing = required.difference(query)
        if missing:
            raise ValueError(f"Query at index {i} is missing fields: {sorted(missing)}")
        if not query["question"].strip():
            raise ValueError(f"Query {query['id']} has an empty question.")
        if not query["relevant_doc_ids"]:
            raise ValueError(f"Query {query['id']} has no relevant_doc_ids.")

    return queries


def ranked_doc_ids_from_passages(passages: Sequence[dict[str, Any]]) -> list[str]:
    """Convert ranked retrieved chunks/passages into ranked document ids."""

    doc_ids: list[str] = []
    seen = set()
    for passage in passages:
        doc_id = passage.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def check_against_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return a list of human-readable failures where a metric is below baseline."""
    failures: list[str] = []
    for metric, minimum in baseline.items():
        if metric.startswith("_") or not isinstance(minimum, (int, float)):
            continue
        actual = summary.get(metric)
        if actual is None:
            failures.append(f"{metric}: missing from summary")
        elif actual < minimum:
            failures.append(f"{metric}: {actual:.3f} < required {minimum:.3f}")
    return failures


def run_retrieval_evaluation(
    *,
    queries_path: str | Path = DEFAULT_QUERIES,
    output_path: str | Path | None = DEFAULT_OUTPUT,
    k: int = 3,
    base_k: int = 8,
    top_n: int = 6,
    expand_hops: int = 1,
) -> dict[str, Any]:
    """Run the golden-query retrieval evaluation against the current index."""

    # Import lazily so metric tests do not require FAISS or sentence-transformers.
    from backend.retriever import expand_and_rerank, get_store

    queries = load_golden_queries(queries_path)
    store = get_store()
    evaluations = []

    for query in queries:
        passages = expand_and_rerank(
            query["question"],
            base_k=base_k,
            expand_hops=expand_hops,
            top_n=top_n,
            store=store,
        )
        retrieved_doc_ids = ranked_doc_ids_from_passages(passages)
        evaluations.append(
            evaluate_query(
                query_id=query["id"],
                question=query["question"],
                relevant_doc_ids=query["relevant_doc_ids"],
                retrieved_doc_ids=retrieved_doc_ids,
                k=k,
            )
        )

    report = {
        "config": {
            "k": k,
            "base_k": base_k,
            "top_n": top_n,
            "expand_hops": expand_hops,
            "queries_path": str(Path(queries_path)),
        },
        "summary": summarize_evaluations(evaluations),
        "queries": [evaluation.to_dict() for evaluation in evaluations],
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on golden queries.")
    parser.add_argument(
        "--queries", default=str(DEFAULT_QUERIES), help="Path to golden queries JSON."
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Where to write the JSON report."
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Rank cutoff for hit/precision/recall metrics."
    )
    parser.add_argument(
        "--base-k", type=int, default=8, help="Initial vector-search candidate count."
    )
    parser.add_argument("--top-n", type=int, default=6, help="Number of reranked chunks to keep.")
    parser.add_argument("--expand-hops", type=int, default=1, help="Concept-graph expansion hops.")
    parser.add_argument(
        "--fail-under",
        default=None,
        help="Path to a baseline JSON; exit non-zero if any metric falls below it.",
    )
    args = parser.parse_args()

    report = run_retrieval_evaluation(
        queries_path=args.queries,
        output_path=args.output,
        k=args.k,
        base_k=args.base_k,
        top_n=args.top_n,
        expand_hops=args.expand_hops,
    )
    summary = report["summary"]
    print(
        "Retrieval evaluation complete: "
        f"queries={int(summary['num_queries'])}, "
        f"hit@{args.k}={summary['mean_hit_at_k']:.3f}, "
        f"precision@{args.k}={summary['mean_precision_at_k']:.3f}, "
        f"recall@{args.k}={summary['mean_recall_at_k']:.3f}, "
        f"MRR={summary['mean_reciprocal_rank']:.3f}, "
        f"nDCG@{args.k}={summary['mean_ndcg_at_k']:.3f}"
    )
    print(f"Report written to {args.output}")

    if args.fail_under:
        with open(args.fail_under, encoding="utf-8") as f:
            baseline = json.load(f)
        failures = check_against_baseline(summary, baseline)
        if failures:
            raise SystemExit("Retrieval quality regressed:\n  " + "\n  ".join(failures))
        print(f"All metrics meet the baseline in {args.fail_under}.")


if __name__ == "__main__":
    main()
