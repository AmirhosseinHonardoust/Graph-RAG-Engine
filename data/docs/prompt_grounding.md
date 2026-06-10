# Prompt Grounding

When an optional LLM is used, the prompt should instruct the model to answer only from retrieved context. If the context is insufficient, the model should say that the corpus does not contain enough information.

Prompt grounding reduces hallucination risk but does not remove it. The application should keep citations visible and avoid presenting generated answers as verified facts.

A fallback path is useful. If no API key is configured or an LLM request fails, the system can return an extractive answer based on retrieved passages.

The UI should clearly show whether the current answer came from extractive mode or LLM mode.
