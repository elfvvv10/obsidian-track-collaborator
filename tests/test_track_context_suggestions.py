"""Tests for assistant-suggested Track Context updates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import QueryRequest, TrackContext, TrackContextSuggestions
from services.query_service import QueryService
from services.track_context_service import TrackContextService
from services.track_context_suggestion_service import TrackContextSuggestionService
from utils import RetrievedChunk


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


class TrackContextSuggestionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TrackContextSuggestionService()

    def test_no_yaml_track_context_returns_none(self) -> None:
        self.assertIsNone(self.service.suggest("Issue: drop lacks contrast", None))

    def test_conservative_extraction_returns_small_suggestion_set(self) -> None:
        suggestions = self.service.suggest(
            "\n".join(
                [
                    "Issue: drop lacks contrast",
                    "Goal: increase build tension before first drop",
                    "Current problem: bassline groove may need more syncopation",
                    "This feels like an arrangement problem.",
                ]
            ),
            TrackContext(track_id="moonlit_driver"),
        )

        self.assertIsNotNone(suggestions)
        self.assertEqual(suggestions.known_issues, ["drop lacks contrast"])
        self.assertEqual(suggestions.goals, ["increase build tension before first drop"])
        self.assertEqual(suggestions.current_stage, "arrangement")
        self.assertEqual(suggestions.current_problem, "bassline groove may need more syncopation")

    def test_avoids_duplicates_and_returns_none_when_nothing_new_exists(self) -> None:
        suggestions = self.service.suggest(
            "Issue: drop lacks contrast\nGoal: increase build tension before first drop",
            TrackContext(
                track_id="moonlit_driver",
                known_issues=["drop lacks contrast"],
                goals=["increase build tension before first drop"],
            ),
        )

        self.assertIsNone(suggestions)


class TrackContextApplySuggestionTests(unittest.TestCase):
    def test_apply_suggestions_appends_unique_items_and_replaces_non_empty_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackContextService(make_config(root))
            service.update_fields(
                "moonlit_driver",
                {
                    "known_issues": ["drop lacks contrast"],
                    "goals": ["finish arrangement"],
                    "current_stage": "writing",
                    "current_problem": "intro energy is too static",
                },
            )

            updated = service.apply_suggestions(
                "moonlit_driver",
                TrackContextSuggestions(
                    known_issues=["drop lacks contrast", "bass lacks movement"],
                    goals=["finish arrangement", "increase build tension"],
                    current_stage="arrangement",
                    current_problem="bassline groove may need more syncopation",
                ),
            )

            self.assertEqual(updated.known_issues, ["drop lacks contrast", "bass lacks movement"])
            self.assertEqual(updated.goals, ["finish arrangement", "increase build tension"])
            self.assertEqual(updated.current_stage, "arrangement")
            self.assertEqual(updated.current_problem, "bassline groove may need more syncopation")


class QueryServiceTrackContextSuggestionTests(unittest.TestCase):
    def test_suggestions_attached_only_when_yaml_track_context_is_active(self) -> None:
        tracking: dict[str, object] = {"saved": False}

        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload):
                return "Issue: drop lacks contrast\nGoal: increase build tension before first drop"

        class StubRetriever:
            def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
                pass

            def retrieve(self, query: str, filters=None, options=None, retrieval_scope=None):
                return [
                    RetrievedChunk(
                        text="Track note",
                        metadata={"note_title": "Track Note", "source_path": "track.md"},
                        distance_or_score=0.1,
                    )
                ]

        class StubVectorStore:
            def __init__(self, config: AppConfig) -> None:
                pass

            def is_index_compatible(self) -> bool:
                return True

            def count(self) -> int:
                return 1

        class StubWebSearchService:
            def __init__(self, config: AppConfig) -> None:
                pass

            def search(self, query: str):
                return []

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = QueryService(
                make_config(root),
                embedding_client_cls=StubEmbeddingClient,
                chat_client_cls=StubChatClient,
                retriever_cls=StubRetriever,
                vector_store_cls=StubVectorStore,
                web_search_service_cls=StubWebSearchService,
                capture_debug_trace=False,
            )

            response_with_context = service.ask(
                QueryRequest(
                    question="Critique this drop",
                    track_id="moonlit_driver",
                    use_track_context=True,
                    track_context=TrackContext(track_id="moonlit_driver"),
                )
            )
            response_without_context = service.ask(QueryRequest(question="Critique this drop", use_track_context=False))

            self.assertEqual(
                response_with_context.answer,
                "Issue: drop lacks contrast\nGoal: increase build tension before first drop\n\nEvidence used: [Local 1]",
            )
            self.assertIsNotNone(response_with_context.track_context_suggestions)
            self.assertIsNone(response_without_context.track_context_suggestions)
            self.assertFalse(tracking["saved"])
