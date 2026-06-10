# FastAPI Backend

FastAPI is useful for exposing RAG functionality through typed HTTP endpoints. A simple backend can provide health checks, question-answering routes, and document recommendation routes.

For this project, the backend should stay lightweight at import time. Expensive retrieval artifacts such as FAISS indexes, embedding models, and graph stores should be loaded lazily only when a retrieval endpoint needs them.

Typed request and response contracts make the API easier to test. The `/ask` endpoint can accept a question and answer mode, then return an answer, passages, citations, graph paths, and retrieval traces.

A production backend would add authentication, rate limits, request logging, structured errors, tracing, and deployment configuration. The demo backend focuses on clarity and testability.
