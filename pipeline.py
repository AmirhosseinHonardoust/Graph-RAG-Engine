from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "evaluation" / "baseline.json"


def main() -> None:
    """Build the retrieval index, then evaluate it against the baseline.

    Heavy dependencies are imported lazily so importing this module (e.g. for the
    console-script entry point) does not require FAISS or sentence-transformers.
    """
    from evaluation.evaluate_retrieval import (
        DEFAULT_OUTPUT,
        check_against_baseline,
        run_retrieval_evaluation,
    )
    from ingest.ingest_docs import DOCS_DIR, build_index, load_docs

    docs = load_docs()
    if not docs:
        raise SystemExit(f"No docs found in {DOCS_DIR}")
    build_index(docs)
    print(f"Ingested {len(docs)} docs.")

    report = run_retrieval_evaluation(output_path=DEFAULT_OUTPUT)
    summary = report["summary"]
    print(
        "Evaluation: " + ", ".join(f"{k}={v:.3f}" for k, v in summary.items() if k != "num_queries")
    )

    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        failures = check_against_baseline(summary, baseline)
        if failures:
            raise SystemExit("Retrieval quality regressed:\n  " + "\n  ".join(failures))

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
