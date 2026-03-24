"""Tests for answer-mode policy behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import QueryRequest
from services.query_service import QueryService
from utils import RetrievedChunk
from web_search import WebSearchResult


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="hermes3",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
    )


def make_query_service(
    *,
    local_chunks: list[RetrievedChunk],
    web_results: list[WebSearchResult],
    answer_text: str,
) -> tuple[QueryService, dict[str, object]]:
    tracking: dict[str, object] = {"chat_calls": 0, "last_prompt": None}

    class StubEmbeddingClient:
        def __init__(self, config: AppConfig) -> None:
            pass

    class StubChatClient:
        def __init__(self, config: AppConfig) -> None:
            pass

        def answer_with_prompt(self, prompt_payload):
            tracking["chat_calls"] += 1
            tracking["last_prompt"] = prompt_payload
            return answer_text

    class StubRetriever:
        def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
            pass

        def retrieve(self, query: str, filters=None, options=None):
            return list(local_chunks)

    class StubVectorStore:
        def __init__(self, config: AppConfig) -> None:
            pass

        def is_index_compatible(self) -> bool:
            return True

        def count(self) -> int:
            return max(1, len(local_chunks))

    class StubWebSearchService:
        def __init__(self, config: AppConfig) -> None:
            pass

        def search(self, query: str) -> list[WebSearchResult]:
            return list(web_results)

    root = Path(tempfile.mkdtemp())
    (root / "vault").mkdir()
    (root / "output").mkdir()
    config = make_config(root)
    service = QueryService(
        config,
        embedding_client_cls=StubEmbeddingClient,
        chat_client_cls=StubChatClient,
        retriever_cls=StubRetriever,
        vector_store_cls=StubVectorStore,
        web_search_service_cls=StubWebSearchService,
        capture_debug_trace=False,
    )
    return service, tracking


class AnswerModePolicyTests(unittest.TestCase):
    def test_strict_mode_refuses_when_evidence_is_weak(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Possibly related note",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.95,
                )
            ],
            web_results=[],
            answer_text="This should not be used.",
        )

        response = service.ask(QueryRequest(question="agents?", answer_mode="strict"))

        self.assertEqual(tracking["chat_calls"], 0)
        self.assertIn("Insufficient evidence", response.answer)
        self.assertIn("strict", response.answer_mode_used.value)
        self.assertTrue(any("Strict mode limited the answer" in warning for warning in response.warnings))

    def test_balanced_mode_allows_limited_synthesis(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use retrieval to ground answers.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Agents use retrieval to ground answers [Local 1].\n\n[Inference] This suggests grounding improves trust.",
        )

        response = service.ask(QueryRequest(question="agents?", answer_mode="balanced"))

        self.assertEqual(tracking["chat_calls"], 1)
        self.assertTrue(response.inference_used)
        self.assertEqual(response.answer_mode_used.value, "balanced")
        self.assertIn("[Local 1]", response.answer)

    def test_exploratory_mode_carries_inference_and_web_labels(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use tools.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[WebSearchResult(title="Agents Update", url="https://example.com", snippet="Recent agents context")],
            answer_text="[Local 1] says agents use tools. [Web 1] adds recent agents context.\n\n[Inference] Together they suggest a broader workflow.",
        )

        response = service.ask(
            QueryRequest(question="agents?", answer_mode="exploratory", retrieval_mode="hybrid")
        )

        self.assertEqual(tracking["chat_calls"], 1)
        self.assertIn("[Local 1] Agents (agents.md)", response.sources)
        self.assertIn("[Web 1] Agents Update (https://example.com)", response.sources)
        self.assertEqual(response.debug.evidence_types_used, ("local_note", "web"))
        self.assertTrue(response.debug.inference_used)
        self.assertEqual(response.debug.answer_mode_used.value, "exploratory")

    def test_answer_mode_flows_into_prompt_payload(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use retrieval.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(QueryRequest(question="agents?", answer_mode="strict"))

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.answer_mode.value, "strict")
        self.assertIn("Strict mode instructions", prompt_payload.user_prompt)
        self.assertEqual(response.debug.answer_mode_requested.value, "strict")
        self.assertEqual(response.debug.answer_mode_used.value, "strict")
