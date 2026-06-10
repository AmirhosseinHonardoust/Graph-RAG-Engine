# Local Development and CI

Local development should use the same commands that CI runs. This makes failures easier to reproduce and reduces the chance that the repository only works on one machine.

A useful Graph-RAG CI workflow can compile Python files, run unit tests, rebuild the index, run retrieval evaluation, validate outputs, and upload generated artifacts.

Optional dependencies should be handled carefully. Tests that do not require FAISS or sentence-transformers should still run, while end-to-end retrieval checks can run in CI after installing full requirements.

Clear commands in the README help contributors reproduce ingestion, backend startup, UI launch, and evaluation locally.
