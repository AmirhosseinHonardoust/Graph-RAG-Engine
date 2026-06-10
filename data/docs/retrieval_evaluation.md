# Retrieval Evaluation

Retrieval evaluation checks whether the system returns relevant documents for known questions. A golden-query file usually stores question IDs, query text, relevant document IDs, and notes.

Common metrics include hit@k, precision@k, recall@k, and mean reciprocal rank. These metrics measure different retrieval behaviors.

Hit@k asks whether any relevant document appears in the top results. Recall@k measures how many relevant documents were found. MRR rewards systems that rank the first relevant result higher.

Evaluation should run in CI when possible. This helps catch broken ingestion, missing FAISS artifacts, changed document IDs, or ranking regressions.
