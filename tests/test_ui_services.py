"""Tests for UI-facing service responses and status helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import AppConfig
from retriever import Retriever
from services.index_service import IndexService
from services.models import CollaborationWorkflow, QueryRequest, QueryResponse, RetrievalMode, RetrievalScope
from services.query_service import QueryService
from utils import AnswerResult, RetrievalOptions, RetrievedChunk


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
    def test_query_request_coerces_retrieval_mode(self) -> None:
        request = QueryRequest(
            question="test",
            retrieval_mode="hybrid",
            retrieval_scope="extended",
            collaboration_workflow="sound_design_brainstorm",
        )
        self.assertEqual(request.retrieval_mode, RetrievalMode.HYBRID)
        self.assertEqual(request.retrieval_scope, RetrievalScope.EXTENDED)
        self.assertEqual(request.collaboration_workflow, CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM)

    def test_query_service_returns_debug_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (vault / "knowledge").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            (vault / "knowledge" / "agents.md").write_text(
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
                "services.query_service.OllamaChatClient.answer_with_prompt",
                return_value="Grounded answer",
            ):
                response = QueryService(config).ask(
                    QueryRequest(
                        question="What do my notes say about agents?",
                        options=RetrievalOptions(top_k=1, candidate_count=1, rerank=True),
                    )
                )

            self.assertIn("Grounded answer", response.answer)
            self.assertIn("Evidence used: [Ref 1]", response.answer)
            self.assertEqual(len(response.debug.initial_candidates), 1)
            self.assertEqual(len(response.debug.primary_chunks), 1)
            self.assertTrue(response.debug.reranking_applied)
            self.assertEqual(len(response.retrieved_chunks), 1)
            self.assertEqual(response.debug.retrieval_scope_requested, RetrievalScope.KNOWLEDGE)
            self.assertEqual(response.debug.web_query_strategy.value, "raw_question")
            self.assertEqual(response.debug.web_results_filtered_count, 0)
            self.assertEqual(response.debug.web_failure_reason, "")
            self.assertEqual(response.debug.web_attempts, [])
            self.assertFalse(response.debug.web_retry_used)
            self.assertEqual(response.debug.curated_knowledge_chunks, 1)
            self.assertEqual(response.debug.imported_knowledge_chunks, 0)

    def test_query_service_save_preserves_existing_evidence_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            output_path = root / "output"
            output_path.mkdir()
            config = make_config(root)
            existing = QueryResponse(
                answer_result=AnswerResult(
                    answer="Grounded answer",
                    sources=["[Local] Agents (agents.md)", "[Web] Example (https://example.com)"],
                    retrieved_chunks=[
                        RetrievedChunk(
                            text="Agent note content",
                            metadata={"note_title": "Agents", "source_path": "agents.md"},
                            distance_or_score=0.1,
                        )
                    ],
                ),
                warnings=["Local retrieval may be weak; external web evidence was used to supplement the answer."],
            )

            saved = QueryService(config).save(
                "What do my notes say about agents?",
                existing.answer_result,
                existing_response=existing,
            )

            self.assertIsNotNone(saved.saved_path)
            self.assertEqual(saved.sources, existing.sources)
            self.assertEqual(saved.warnings, existing.warnings)
            self.assertIn("answers/General Asks", str(saved.saved_path))

    def test_retriever_returns_public_debug_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            class StubEmbeddingClient:
                def embed_text(self, text: str) -> list[float]:
                    return [1.0, 0.0]

            class StubVectorStore:
                def count(self) -> int:
                    return 2

                def query(self, query_embedding: list[float], top_k: int, filters=None) -> list[RetrievedChunk]:
                    return [
                        RetrievedChunk("one", {"note_title": "One", "source_path": "one.md"}, 0.1),
                        RetrievedChunk("two", {"note_title": "Two", "source_path": "two.md"}, 0.2),
                    ]

                def get_chunks_by_note_keys(self, note_keys, *, max_chunks_per_note, excluded_note_keys=None):
                    return []

            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())
            debug = retriever.retrieve_with_debug(
                "question",
                options=RetrievalOptions(top_k=1, candidate_count=2, rerank=True),
            )

            self.assertEqual(len(debug.initial_candidates), 2)
            self.assertEqual(len(debug.primary_chunks), 1)
            self.assertEqual(len(debug.final_chunks), 1)

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
