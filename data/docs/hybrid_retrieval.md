# Hybrid Retrieval

Hybrid retrieval combines multiple ranking signals. A Graph-RAG system may use embedding similarity, concept overlap, and graph PageRank to rank candidate passages.

Embedding similarity captures semantic closeness. Concept overlap rewards passages that share important query terms. PageRank can slightly prefer documents that are central in the graph.

The scoring formula should be transparent. Returning a retrieval trace with score components makes the system easier to debug and explain.

Weights should not be trusted blindly. They should be evaluated against golden queries and adjusted only when metrics improve.
