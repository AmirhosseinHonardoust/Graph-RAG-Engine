# TF-IDF vs Embeddings

TF-IDF represents text with sparse lexical features. It works well when the same words appear in both the question and the document, and it is simple to inspect.

Dense embeddings represent text with continuous vectors. They can retrieve semantically related passages even when the exact words differ, which makes them useful for modern RAG retrieval.

A robust retrieval system can combine lexical search, dense vector search, graph expansion, and reranking. Each method captures a different retrieval signal.

TF-IDF is easier to explain, while embeddings are often stronger for semantic search. The best choice depends on corpus size, query style, latency, and evaluation results.
