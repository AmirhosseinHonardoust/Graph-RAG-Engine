# Vector Indexing Tradeoffs

Vector indexing choices affect latency, memory use, and retrieval accuracy. Exact indexes are simple and reliable for small corpora, while approximate indexes are better for larger collections.

`IndexFlatIP` performs exact inner-product search. It is a good choice for small demos with normalized embeddings because it is deterministic and easy to inspect.

Larger systems may use inverted files, HNSW, product quantization, or managed vector databases. Those systems require more tuning and operational monitoring.

Whatever index is used, vector dimensions must match the embedding model. A mismatch usually indicates that the index was generated with a different model version.
