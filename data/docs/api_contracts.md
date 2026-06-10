# API Contracts

API contracts define what a client can send and what it can expect in return. For a RAG backend, contracts are important because the UI, tests, and future integrations depend on stable response fields.

An answer response should include the answer text, the selected answer mode, retrieved passages, citations, and optional warnings. If retrieval tracing is enabled, score components can be returned as structured metadata.

Contract tests help prevent accidental breaking changes. If a field is renamed or removed, tests should fail before the Streamlit UI or external client breaks.

Contracts should avoid leaking internal implementation details that are not useful to users. The goal is to expose enough evidence for review without making the response hard to consume.
