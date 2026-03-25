"""Tests for provider-aware model client construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient, OpenAIChatClient
from model_provider import create_chat_client, create_embedding_client


def make_config() -> AppConfig:
    return AppConfig(
        obsidian_vault_path=Path(tempfile.gettempdir()),
        obsidian_output_path=Path(tempfile.gettempdir()),
        chroma_db_path=Path(tempfile.gettempdir()),
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="deepseek",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
    )


class ModelProviderTests(unittest.TestCase):
    def test_create_chat_client_uses_ollama_by_default(self) -> None:
        client = create_chat_client(make_config())
        self.assertIsInstance(client, OllamaChatClient)

    def test_create_embedding_client_uses_ollama_by_default(self) -> None:
        client = create_embedding_client(make_config())
        self.assertIsInstance(client, OllamaEmbeddingClient)

    def test_create_chat_client_applies_model_override_to_stub(self) -> None:
        class StubChatClient:
            def __init__(self, config: AppConfig) -> None:
                self.model = config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload) -> str:
                return "ok"

        client = create_chat_client(
            make_config(),
            model_override="deepseek-r1:latest",
            client_cls=StubChatClient,
        )

        self.assertEqual(client.model, "deepseek-r1:latest")

    def test_openai_provider_is_not_implemented_yet(self) -> None:
        config = make_config()
        config.chat_provider = "openai"
        config.openai_api_key = "test-key"
        config.openai_chat_model = "gpt-4o-mini"
        client = create_chat_client(config)
        self.assertIsInstance(client, OpenAIChatClient)

    def test_openai_provider_requires_api_key(self) -> None:
        config = make_config()
        config.chat_provider = "openai"
        config.openai_chat_model = "gpt-4o-mini"
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            create_chat_client(config)
