# Reranking and PageRank

Reranking adjusts the order of candidate passages after initial retrieval. In Graph-RAG, candidates can come from vector search and graph expansion.

A transparent reranker can combine embedding similarity, normalized concept overlap, and document PageRank. Each component should be exposed in a retrieval trace.

PageRank should usually be a small signal. If it dominates the ranking, central documents may appear too often even when they are not the most relevant answer source.

Reranking should be tested with golden queries. If adding PageRank or graph expansion lowers retrieval quality, the weights should be changed.
