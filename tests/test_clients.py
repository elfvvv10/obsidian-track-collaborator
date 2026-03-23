"""Mocked tests for Ollama clients."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from requests.exceptions import RequestException

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient


def make_config() -> AppConfig:
    config = AppConfig(
        obsidian_vault_path=Path(tempfile.gettempdir()),
        obsidian_output_path=Path(tempfile.gettempdir()),
        chroma_db_path=Path(tempfile.gettempdir()),
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="hermes3",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
    )
    return config


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class OllamaEmbeddingClientTests(unittest.TestCase):
    def test_embed_texts_success(self) -> None:
        client = OllamaEmbeddingClient(make_config())

        with patch(
            "embeddings.requests.request",
            side_effect=[
                FakeResponse(200, {"models": [{"name": "nomic-embed-text"}]}),
                FakeResponse(200, {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}),
            ],
        ) as request_mock:
            result = client.embed_texts(["hello", "world"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(request_mock.call_count, 2)

    def test_embed_texts_missing_model(self) -> None:
        client = OllamaEmbeddingClient(make_config())

        with patch(
            "embeddings.requests.request",
            return_value=FakeResponse(200, {"models": [{"name": "other-model"}]}),
        ):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                client.embed_texts(["hello"])

    def test_embed_texts_handles_unreachable_ollama(self) -> None:
        client = OllamaEmbeddingClient(make_config())

        with patch("embeddings.requests.request", side_effect=RequestException("boom")):
            with self.assertRaisesRegex(RuntimeError, "Could not reach Ollama"):
                client.embed_texts(["hello"])


class OllamaChatClientTests(unittest.TestCase):
    def test_answer_question_success(self) -> None:
        client = OllamaChatClient(make_config())

        with patch(
            "llm.requests.request",
            side_effect=[
                FakeResponse(200, {"models": [{"name": "hermes3"}]}),
                FakeResponse(200, {"message": {"content": "Grounded answer"}}),
            ],
        ):
            answer = client.answer_question("What is this?", [])

        self.assertEqual(answer, "Grounded answer")

    def test_answer_question_empty_response(self) -> None:
        client = OllamaChatClient(make_config())

        with patch(
            "llm.requests.request",
            side_effect=[
                FakeResponse(200, {"models": [{"name": "hermes3"}]}),
                FakeResponse(200, {"message": {"content": ""}}),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "empty chat response"):
                client.answer_question("What is this?", [])
