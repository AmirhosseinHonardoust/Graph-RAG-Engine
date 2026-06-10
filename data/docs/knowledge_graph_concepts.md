# Knowledge Graph Concepts

A lightweight knowledge graph can connect chunks through extracted concepts. Concepts may be keywords, entities, technical terms, or domain labels.

Concept extraction does not need to be perfect for a demo, but it should be deterministic and understandable. Simple token-based concept extraction is easy to test, while entity extraction can be added later.

Concept nodes support graph expansion. If two chunks mention the same concept, the system can discover them as neighbors even when they come from different documents.

Graph explanations should show shared concepts and paths so users can understand why extra passages were retrieved.
