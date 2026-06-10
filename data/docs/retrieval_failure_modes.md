# Retrieval Failure Modes

RAG systems can fail in several ways. The retriever may return irrelevant chunks, miss the best document, over-rank a central document, or return no useful context.

Failures can come from poor chunking, weak embeddings, stale indexes, ambiguous queries, bad reranking weights, or missing source documents. Golden queries help expose these problems early.

The answer layer should handle weak retrieval gracefully. If the context is not enough, it is safer to say that the corpus does not contain enough information than to invent an answer.

Failure analysis should inspect the retrieved passages, graph expansion paths, and retrieval trace score components rather than only looking at the final answer.
