# Citation Grounding

Grounded RAG systems should show where an answer came from. Citations help users inspect source passages and reduce blind trust in generated text.

An extractive answer can quote or summarize retrieved passages directly. An LLM answer should still be constrained by retrieved context and should expose citations.

Good citation metadata includes document ID, title, source path, chunk ID, and snippet text. Repo-relative paths are better than local machine paths because they are portable.

Citations do not guarantee truth. They show evidence from the indexed corpus, so corpus quality and retrieval quality still matter.
