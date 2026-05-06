"""Tests for assistant-suggested Track Context updates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import QueryRequest, TrackContext, TrackContextSuggestions, TrackContextUpdateProposal
from services.query_service import QueryService
from services.track_context_service import TrackContextService
from services.track_context_suggestion_service import TrackContextSuggestionService
from services.track_context_update_service import TrackContextUpdateService
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

    def test_extracts_new_track_context_suggestion_fields_cleanly(self) -> None:
        suggestions = self.service.suggest(
            "\n".join(
                [
                    "Vibe: dark and driving",
                    "Reference track: Bicep - Glue",
                    "Tempo: 126",
                    "key of F#m",
                    "Focus on the drop",
                ]
            ),
            TrackContext(track_id="moonlit_driver"),
        )

        self.assertIsNotNone(suggestions)
        self.assertEqual(suggestions.vibe_suggestions, ["dark and driving"])
        self.assertEqual(suggestions.reference_track_suggestions, ["Bicep - Glue"])
        self.assertEqual(suggestions.bpm_suggestion, 126)
        self.assertEqual(suggestions.key_suggestion, "F#m")
        self.assertEqual(suggestions.section_focus, "drop")

    def test_new_track_context_suggestions_stay_conservative(self) -> None:
        suggestions = self.service.suggest(
            "\n".join(
                [
                    "Vibe: dark and driving",
                    "Reference track: Bicep - Glue",
                    "Tempo: 999",
                    "I like the kick but it needs less mud.",
                    "Focus on the main groove",
                ]
            ),
            TrackContext(
                track_id="moonlit_driver",
                vibe=["dark and driving"],
                reference_tracks=["Bicep - Glue"],
            ),
        )

        self.assertIsNotNone(suggestions)
        self.assertEqual(suggestions.vibe_suggestions, [])
        self.assertEqual(suggestions.reference_track_suggestions, [])
        self.assertIsNone(suggestions.bpm_suggestion)
        self.assertEqual(suggestions.section_focus, "main groove")


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

    def test_apply_suggestions_merges_new_context_fields_after_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackContextService(make_config(root))
            service.update_fields(
                "moonlit_driver",
                {
                    "vibe": ["dark"],
                    "reference_tracks": ["Bicep - Glue"],
                    "bpm": 124,
                    "key": "A minor",
                },
            )

            updated = service.apply_suggestions(
                "moonlit_driver",
                TrackContextSuggestions(
                    vibe_suggestions=["dark", "driving"],
                    reference_track_suggestions=["Bicep - Glue", "Floating Points - Ratio"],
                    bpm_suggestion=126,
                    key_suggestion="F#m",
                ),
            )

            self.assertEqual(updated.vibe, ["dark", "driving"])
            self.assertEqual(updated.reference_tracks, ["Bicep - Glue", "Floating Points - Ratio"])
            self.assertEqual(updated.bpm, 126)
            self.assertEqual(updated.key, "F#m")


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


class TrackContextUpdateServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TrackContextUpdateService()

    def test_extract_returns_cleaned_answer_and_structured_proposal(self) -> None:
        answer = (
            "The drop needs a clearer payoff.\n\n"
            "```track_context_update\n"
            '{'
            '"track_id": "warehouse-hypnosis-01", '
            '"summary": "Capture the main drop issue.", '
            '"set_fields": {"genre": "peak-time hypnotic techno"}, '
            '"add_to_lists": {"current_issues": ["drop has no payoff"]}, '
            '"remove_from_lists": {}, '
            '"confidence": "high", '
            '"source_reasoning": "The user explicitly reframed the track genre and described the drop problem."'
            "}\n"
            "```"
        )

        cleaned, proposal = self.service.extract(
            answer,
            TrackContext(track_id="warehouse-hypnosis-01"),
        )

        self.assertEqual(cleaned, "The drop needs a clearer payoff.")
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.track_id, "warehouse-hypnosis-01")
        self.assertEqual(proposal.set_fields["genre"], "peak-time hypnotic techno")
        self.assertEqual(proposal.add_to_lists["current_issues"], ["drop has no payoff"])

    def test_apply_updates_scalars_and_deduplicates_list_additions(self) -> None:
        updated = self.service.apply(
            TrackContext(
                track_id="warehouse-hypnosis-01",
                track_name="Warehouse Hypnosis",
                reference_tracks=["boris-brejcha-gravity"],
                known_issues=["drop has no payoff"],
            ),
            TrackContextUpdateProposal(
                track_id="warehouse-hypnosis-01",
                set_fields={"title": "Warehouse Hypnosis v2"},
                add_to_lists={
                    "references": ["boris-brejcha-gravity", "enrico-sangiuliano-moon-rock"],
                    "current_issues": ["drop has no payoff", "build lacks tension"],
                    "next_actions": ["test a simpler offbeat hat pattern"],
                },
            ),
        )

        self.assertEqual(updated.track_name, "Warehouse Hypnosis v2")
        self.assertEqual(
            updated.reference_tracks,
            ["boris-brejcha-gravity", "enrico-sangiuliano-moon-rock"],
        )
        self.assertEqual(updated.known_issues, ["drop has no payoff", "build lacks tension"])
        self.assertEqual(updated.goals, ["test a simpler offbeat hat pattern"])

    def test_apply_ignores_malformed_or_mismatched_updates_safely(self) -> None:
        base = TrackContext(track_id="warehouse-hypnosis-01", track_name="Warehouse Hypnosis")

        unchanged = self.service.apply(
            base,
            TrackContextUpdateProposal(
                track_id="other-track",
                set_fields={"title": ""},
                add_to_lists={"references": [""]},
            ),
        )

        self.assertEqual(unchanged, base)

    def test_apply_can_set_sections_and_add_section_issues(self) -> None:
        updated = self.service.apply(
            TrackContext(
                track_id="warehouse-hypnosis-01",
                track_name="Warehouse Hypnosis",
            ),
            TrackContextUpdateProposal(
                track_id="warehouse-hypnosis-01",
                set_sections={
                    "drop": {
                        "role": "main groove",
                        "energy_level": "high",
                        "elements": ["kick", "bass", "stab"],
                    }
                },
                add_section_issues={
                    "drop": ["no new motif introduced"],
                },
            ),
        )

        self.assertIn("drop", updated.sections)
        self.assertEqual(updated.sections["drop"].role, "main groove")
        self.assertEqual(updated.sections["drop"].energy_level, "high")
        self.assertEqual(updated.sections["drop"].elements, ["kick", "bass", "stab"])
        self.assertEqual(updated.sections["drop"].issues, ["no new motif introduced"])

    def test_apply_can_add_section_elements_and_notes_without_clobbering_existing_values(self) -> None:
        updated = self.service.apply(
            TrackContext(
                track_id="warehouse-hypnosis-01",
                sections={
                    "drop": {
                        "name": "drop",
                        "elements": ["kick", "bass"],
                        "notes": "needs stronger anchor",
                    }
                },
            ),
            TrackContextUpdateProposal(
                track_id="warehouse-hypnosis-01",
                add_section_elements={
                    "drop": ["bass", "stab", "ride"],
                },
                add_section_notes={
                    "drop": [
                        "test a shorter stab rhythm",
                        "needs stronger anchor",
                    ]
                },
            ),
        )

        self.assertEqual(updated.sections["drop"].elements, ["kick", "bass", "stab", "ride"])
        self.assertEqual(
            updated.sections["drop"].notes,
            "needs stronger anchor\ntest a shorter stab rhythm",
        )

    def test_extract_ignores_invalid_json_block_without_crashing(self) -> None:
        cleaned, proposal = self.service.extract(
            "Keep the bassline simpler.\n\n```track_context_update\n{not valid json}\n```",
            TrackContext(track_id="warehouse-hypnosis-01"),
        )

        self.assertEqual(cleaned, "Keep the bassline simpler.")
        self.assertIsNone(proposal)

    def test_extract_accepts_set_active_section_alias_for_session_focus(self) -> None:
        cleaned, proposal = self.service.extract(
            (
                "Let's keep working on the drop.\n\n"
                "```track_context_update\n"
                '{'
                '"track_id": "warehouse-hypnosis-01", '
                '"summary": "Keep the conversation focused on the drop.", '
                '"set_fields": {}, '
                '"add_to_lists": {}, '
                '"remove_from_lists": {}, '
                '"set_active_section": "Drop", '
                '"confidence": "medium", '
                '"source_reasoning": "The user is clearly still talking about the drop."'
                "}\n"
                "```"
            ),
            TrackContext(track_id="warehouse-hypnosis-01"),
        )

        self.assertEqual(cleaned, "Let's keep working on the drop.")
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.section_focus, "drop")


class QueryServiceTrackContextUpdateTests(unittest.TestCase):
    def test_query_response_carries_optional_track_context_update(self) -> None:
        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload):
                return (
                    "Try simplifying the offbeat hat before the drop.\n\n"
                    "```track_context_update\n"
                    '{'
                    '"track_id": "warehouse-hypnosis-01", '
                    '"summary": "Capture the next actions from this turn.", '
                    '"set_fields": {}, '
                    '"add_to_lists": {"next_actions": ["automate reverb send in the fill before the drop"]}, '
                    '"remove_from_lists": {}, '
                    '"confidence": "medium", '
                    '"source_reasoning": "The user asked to capture a concrete next step for the active track."'
                    "}\n"
                    "```"
                )

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

            response = service.ask(
                QueryRequest(
                    question="save that to the track context",
                    track_id="warehouse-hypnosis-01",
                    use_track_context=True,
                    track_context=TrackContext(track_id="warehouse-hypnosis-01"),
                )
            )

            self.assertNotIn("track_context_update", response.answer)
            self.assertIsNotNone(response.track_context_update)
            self.assertTrue(response.track_context_update_available)
            self.assertEqual(
                response.track_context_update.add_to_lists["next_actions"],
                ["automate reverb send in the fill before the drop"],
            )

    def test_query_response_can_use_provider_structured_output_fallback(self) -> None:
        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload):
                return "Try simplifying the offbeat hat before the drop."

            def answer_with_json_schema(self, *, system_prompt, user_prompt, schema_name, json_schema):
                return {
                    "track_id": "warehouse-hypnosis-01",
                    "summary": "Capture the next actions from this turn.",
                    "set_fields": {},
                    "add_to_lists": {
                        "next_actions": ["test a simpler offbeat hat pattern"],
                    },
                    "remove_from_lists": {},
                    "confidence": "medium",
                    "source_reasoning": "The answer ended with a concrete next action for the active track.",
                }

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

            response = service.ask(
                QueryRequest(
                    question="capture that as a next action",
                    track_id="warehouse-hypnosis-01",
                    use_track_context=True,
                    track_context=TrackContext(track_id="warehouse-hypnosis-01"),
                )
            )

            self.assertEqual(response.answer, "Try simplifying the offbeat hat before the drop.\n\nEvidence used: [Local 1]")
            self.assertIsNotNone(response.track_context_update)
            self.assertEqual(
                response.track_context_update.add_to_lists["next_actions"],
                ["test a simpler offbeat hat pattern"],
            )

    def test_query_response_skips_structured_output_when_provider_does_not_support_it(self) -> None:
        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model
                self.provider = "custom"

            def answer_with_prompt(self, prompt_payload):
                return "Try a slightly simpler hat pattern before the drop."

            def answer_with_json_schema(self, *, system_prompt, user_prompt, schema_name, json_schema):
                raise AssertionError("Structured JSON should not be requested for unsupported providers.")

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

            response = service.ask(
                QueryRequest(
                    question="capture that as a next action",
                    track_id="warehouse-hypnosis-01",
                    use_track_context=True,
                    track_context=TrackContext(track_id="warehouse-hypnosis-01"),
                )
            )

            self.assertFalse(response.track_context_update_available)
            self.assertIsNone(response.track_context_update)
