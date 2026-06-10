# Document Metadata

Document metadata gives retrieved passages context. Useful metadata includes document ID, title, source path, chunk ID, and extracted concepts.

Metadata supports citations, graph construction, UI display, evaluation, and debugging. Without metadata, a RAG system can return text but cannot easily explain where it came from.

Metadata should be stable across ingestion runs. Stable IDs make it easier to compare retrieval evaluation results over time.

Sensitive metadata should not be exposed accidentally. Public demos should avoid local user paths, secrets, private file names, or organization-specific identifiers.
