from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful retrieval-augmented assistant. Answer only from the "
    "provided sources. If the sources do not contain enough information, say so. "
    "Cite sources using the bracketed source numbers, such as [1] or [2]."
)


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an OpenAI-compatible chat-completions endpoint.

    The project stays runnable without an API key. LLM mode is optional and is
    activated only when the user explicitly requests it and an API key exists.
    """

    api_key: str | None
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 30
    temperature: float = 0.2
    max_tokens: int = 500

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            api_key=os.getenv("GRAPH_RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            model=os.getenv("GRAPH_RAG_LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("GRAPH_RAG_LLM_BASE_URL", "https://api.openai.com/v1"),
            timeout_seconds=int(os.getenv("GRAPH_RAG_LLM_TIMEOUT_SECONDS", "30")),
            temperature=float(os.getenv("GRAPH_RAG_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("GRAPH_RAG_LLM_MAX_TOKENS", "500")),
        )


def build_source_context(passages: list[dict[str, Any]], max_chars_per_source: int = 1200) -> str:
    """Format retrieved passages as numbered source context for an LLM."""

    if not passages:
        return "No retrieved sources."

    blocks = []
    for idx, passage in enumerate(passages, start=1):
        title = passage.get("doc_title", "Untitled")
        url = passage.get("url", "")
        text = str(passage.get("text", "")).strip()
        if len(text) > max_chars_per_source:
            text = text[: max_chars_per_source - 3].rstrip() + "..."
        blocks.append(f"[{idx}] {title}\nURL: {url}\n{text}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    """Build a grounded RAG prompt from a question and retrieved passages."""

    context = build_source_context(passages)
    return (
        "Answer the question using only the sources below.\n"
        "Requirements:\n"
        "- Be concise and direct.\n"
        "- Use source citations like [1] or [2].\n"
        "- Do not invent facts that are not supported by the sources.\n"
        "- If the sources are insufficient, state what is missing.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Sources:\n{context}\n\n"
        "Answer:"
    )


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def generate_llm_answer(
    question: str,
    passages: list[dict[str, Any]],
    config: LLMConfig | None = None,
) -> str:
    """Generate a grounded answer with an OpenAI-compatible API.

    This function intentionally uses only the Python standard library so the
    project does not need an additional SDK dependency. It raises a clear error
    when LLM mode is requested without credentials.
    """

    config = config or LLMConfig.from_env()
    if not config.api_key:
        raise RuntimeError(
            "LLM mode requested, but no API key was found. Set "
            "GRAPH_RAG_LLM_API_KEY or OPENAI_API_KEY, or use extractive mode."
        )

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": build_rag_prompt(question, passages)},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")

    req = request.Request(
        _chat_completions_url(config.base_url),
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM provider returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach LLM provider: {exc.reason}") from exc

    data = json.loads(response_body)
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM provider returned an unexpected response format.") from exc

    return str(answer).strip()
