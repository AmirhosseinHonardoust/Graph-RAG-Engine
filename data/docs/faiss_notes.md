# FAISS Notes

FAISS is a vector-search library used for nearest-neighbor retrieval over dense embeddings. In a RAG system, documents are split into chunks, embedded into vectors, and searched by similarity when a user asks a question.

For normalized embeddings, inner-product search can approximate cosine similarity. A simple demo can use `IndexFlatIP` because it is exact, transparent, and easy to debug, even though larger systems often need approximate indexes.

FAISS is useful when the project needs fast retrieval over many chunks. The retrieval layer should still return source metadata, chunk IDs, and document titles so answers can be grounded in citations.

A production deployment would normally monitor index freshness, embedding-model versions, vector dimensions, and rebuild behavior whenever documents change.
