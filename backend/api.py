import os
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag import ask
from .retriever import get_store, recommend_similar


def parse_cors_origins(raw: str | None) -> list[str]:
    """Parse a comma-separated origins string into a list, defaulting to ``["*"]``."""
    if not raw:
        return ["*"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


app = FastAPI(title="Graph RAG MVP")

# Configurable via GRAPH_RAG_ALLOW_ORIGINS (comma-separated). Defaults to "*" for
# local development. Credentials are disabled when origins are wildcarded, since
# browsers reject `*` combined with credentialed requests.
_allow_origins = parse_cors_origins(os.getenv("GRAPH_RAG_ALLOW_ORIGINS"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskReq(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["extractive", "llm"] = "extractive"


class RecReq(BaseModel):
    doc_id: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask_ep(payload: AskReq):
    return ask(payload.question, mode=payload.mode)


@app.post("/recommend")
def rec_ep(payload: RecReq):
    return {"items": recommend_similar(payload.doc_id)}


@app.get("/docs_list")
def docs_list():
    store = get_store()
    docs = {}
    for chunk in store.chunks:
        docs[chunk["doc_id"]] = {"title": chunk["doc_title"], "url": chunk["url"]}
    return {"items": [{"doc_id": doc_id, **metadata} for doc_id, metadata in docs.items()]}
