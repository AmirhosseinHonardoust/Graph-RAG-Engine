# Production RAG Considerations

A production RAG system needs more than a working demo. It needs index refresh workflows, observability, prompt and retrieval evaluation, access control, monitoring, and safety review.

Document permissions are important. Users should only retrieve sources they are allowed to see. This demo does not implement permission-aware retrieval.

Operational monitoring should track latency, empty retrieval results, answer-mode failures, index freshness, corpus changes, and evaluation regressions.

This project is best understood as a portfolio MVP that demonstrates Graph-RAG architecture, not a complete enterprise knowledge system.
