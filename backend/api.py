from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag import ask
from .retriever import get_store, recommend_similar

app = FastAPI(title="Graph RAG MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskReq(BaseModel):
    question: str = Field(..., min_length=1)


class RecReq(BaseModel):
    doc_id: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask_ep(payload: AskReq):
    return ask(payload.question)


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
