# Index Refresh Workflow

Whenever documents change, the retrieval index should be rebuilt. This keeps FAISS vectors, chunk metadata, graph artifacts, and evaluation results aligned with the current corpus.

A clean refresh workflow usually runs ingestion first, then retrieval evaluation. The workflow should verify that index files exist and that the golden-query report includes summary metrics.

Committed artifacts should not contain local absolute paths. Source URLs and citation paths should be repository-relative so the project works across machines.

For larger systems, index refresh may be scheduled, triggered by document changes, or run through a deployment pipeline with versioned artifacts.
