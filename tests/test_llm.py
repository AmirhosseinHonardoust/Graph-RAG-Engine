import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.llm import LLMConfig, build_rag_prompt, build_source_context, generate_llm_answer

PASSAGES = [
    {
        "doc_title": "FAISS Notes",
        "url": "https://example.com/faiss",
        "text": "FAISS is a library for efficient vector similarity search.",
    },
    {
        "doc_title": "Streamlit Intro",
        "url": "https://example.com/streamlit",
        "text": "Streamlit can turn Python scripts into interactive data apps.",
    },
]


class LLMUtilityTests(unittest.TestCase):
    def test_build_source_context_numbers_sources(self):
        context = build_source_context(PASSAGES)

        self.assertIn("[1] FAISS Notes", context)
        self.assertIn("[2] Streamlit Intro", context)
        self.assertIn("URL: https://example.com/faiss", context)

    def test_build_rag_prompt_includes_grounding_rules(self):
        prompt = build_rag_prompt("What is FAISS?", PASSAGES)

        self.assertIn("Answer the question using only the sources below", prompt)
        self.assertIn("What is FAISS?", prompt)
        self.assertIn("[1] FAISS Notes", prompt)
        self.assertIn("Do not invent facts", prompt)

    def test_config_reads_graph_rag_environment_first(self):
        env = {
            "GRAPH_RAG_LLM_API_KEY": "graph-key",
            "OPENAI_API_KEY": "openai-key",
            "GRAPH_RAG_LLM_MODEL": "custom-model",
            "GRAPH_RAG_LLM_BASE_URL": "https://llm.example.com/v1",
            "GRAPH_RAG_LLM_TIMEOUT_SECONDS": "7",
            "GRAPH_RAG_LLM_TEMPERATURE": "0.1",
            "GRAPH_RAG_LLM_MAX_TOKENS": "123",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()

        self.assertEqual(config.api_key, "graph-key")
        self.assertEqual(config.model, "custom-model")
        self.assertEqual(config.base_url, "https://llm.example.com/v1")
        self.assertEqual(config.timeout_seconds, 7)
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_tokens, 123)

    def test_generate_llm_answer_requires_key(self):
        with self.assertRaisesRegex(RuntimeError, "no API key"):
            generate_llm_answer("What is FAISS?", PASSAGES, config=LLMConfig(api_key=None))

    def test_source_context_truncates_long_passages(self):
        long_passages = [
            {
                "doc_title": "Long Doc",
                "url": "https://example.com/long",
                "text": "x" * 100,
            }
        ]
        context = build_source_context(long_passages, max_chars_per_source=20)

        self.assertIn("Long Doc", context)
        self.assertIn("...", context)
        self.assertLess(len(context), 90)


if __name__ == "__main__":
    unittest.main()
