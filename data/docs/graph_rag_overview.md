# Graph-RAG Overview

Graph-RAG combines vector retrieval with graph structure. Dense embeddings retrieve semantically similar chunks, while a graph can connect related documents, chunks, and concepts.

The graph layer can help when a query mentions a concept that appears across multiple documents. After vector search finds initial passages, graph expansion can discover neighboring chunks that share concepts.

A simple graph schema can include document nodes, chunk nodes, and concept nodes. Edges can connect documents to chunks and chunks to concepts.

Graph-RAG is not automatically better than vector search. It needs evaluation to prove that graph expansion improves recall or answer grounding for the target corpus.
