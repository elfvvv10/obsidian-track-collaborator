"""Tests for UI-facing service responses and status helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import AppConfig
from services.index_service import IndexService
from services.models import QueryRequest
from services.query_service import QueryService
from utils import RetrievalOptions


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="hermes3",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=2,
    )


class UIFacingServiceTests(unittest.TestCase):
    def test_query_service_returns_debug_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            (vault / "agents.md").write_text(
                "# Agents\n\nAI agents use tools and retrieval.\n",
                encoding="utf-8",
            )

            with patch(
                "services.index_service.OllamaEmbeddingClient.embed_texts",
                return_value=[[1.0, 0.0]],
            ):
                IndexService(config).index(reset_store=True)

            with patch(
                "services.query_service.OllamaEmbeddingClient.embed_text",
                return_value=[1.0, 0.0],
            ), patch(
                "services.query_service.OllamaChatClient.answer_question",
                return_value="Grounded answer",
            ):
                response = QueryService(config).ask(
                    QueryRequest(
                        question="What do my notes say about agents?",
                        options=RetrievalOptions(top_k=1, candidate_count=1, rerank=True),
                    )
                )

            self.assertEqual(response.answer, "Grounded answer")
            self.assertEqual(len(response.debug.initial_candidates), 1)
            self.assertEqual(len(response.debug.primary_chunks), 1)
            self.assertTrue(response.debug.reranking_applied)
            self.assertEqual(len(response.retrieved_chunks), 1)

    def test_index_service_status_reports_paths_and_ollama_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            class StubResponse:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    return {"models": [{"name": "hermes3:latest"}]}

            with patch("services.common.requests.get", return_value=StubResponse()):
                status = IndexService(config).get_status()

            self.assertEqual(status.vault_path, config.obsidian_vault_path)
            self.assertEqual(status.output_path, config.obsidian_output_path)
            self.assertTrue(status.ollama_reachable)
            self.assertIn("hermes3", status.ollama_status_message)
            self.assertFalse(status.ready)
