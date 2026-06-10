# Chunking Strategies

Chunking is the process of splitting documents into smaller passages before embedding. Good chunks preserve enough context to answer questions while staying focused enough for accurate retrieval.

Very small chunks can lose context. Very large chunks can include unrelated information and reduce retrieval precision. A practical system often starts with paragraph-aware chunking and then evaluates retrieval quality.

Chunk metadata is important. Each chunk should keep its document ID, title, source path, chunk ID, and extracted concepts so the answer layer can cite evidence.

Chunking should be deterministic. If the same documents are ingested twice, the system should create stable chunk IDs and stable metadata.
