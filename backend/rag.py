from typing import Any, Dict, List, Literal

from .llm import generate_llm_answer
from .retriever import expand_and_rerank, get_store


def compose_answer_extractive(question: str, passages: List[Dict[str, Any]]) -> str:
    """Compose an extractive answer from retrieved passages.

    This intentionally avoids making unsupported claims. A future LLM mode can
    replace this function while keeping citations and graph paths intact.
    """

    if not passages:
        return "No relevant passages were found in the indexed documents."

    parts = []
    for passage in passages:
        parts.append(
            f"**Source:** [{passage['doc_title']}]({passage['url']})\n> {passage['text']}"
        )
    return "\n\n".join(parts)


def ask(question: str, mode: Literal["extractive", "llm"] = "extractive") -> Dict[str, Any]:
    if mode not in {"extractive", "llm"}:
        raise ValueError("mode must be either 'extractive' or 'llm'")

    store = get_store()
    passages = expand_and_rerank(question, base_k=8, expand_hops=1, top_n=5, store=store)

    llm_error = None
    if mode == "llm":
        try:
            answer = generate_llm_answer(question, passages)
            answer_mode = "llm"
        except Exception as exc:  # Keep the API usable when optional LLM mode fails.
            answer = compose_answer_extractive(question, passages)
            answer_mode = "extractive_fallback"
            llm_error = str(exc)
    else:
        answer = compose_answer_extractive(question, passages)
        answer_mode = "extractive"

    chunk_ids = [passage["id"] for passage in passages]
    paths = store.graph.explain_paths(chunk_ids)
    citations = [
        {"doc_title": passage["doc_title"], "url": passage["url"]}
        for passage in passages
    ]

    response = {
        "answer": answer,
        "answer_mode": answer_mode,
        "citations": citations,
        "paths": paths,
    }
    if llm_error:
        response["llm_error"] = llm_error
    return response
