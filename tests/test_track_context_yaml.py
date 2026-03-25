"""Tests for YAML-backed track context normalization, persistence, and wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from saver import format_track_context_summary, save_answer
from services.models import AnswerMode, CollaborationWorkflow, QueryRequest, RetrievalMode, TrackContext, WorkflowInput
from services.prompt_service import PromptService
from services.query_service import QueryService
from services.track_context_service import TrackContextService
from services.track_context_utils import _clean_dict_str, _clean_list, _clean_str, normalize_track_context
from utils import AnswerResult, RetrievedChunk


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


class TrackContextUtilsTests(unittest.TestCase):
    def test_clean_string_list_and_dict_helpers(self) -> None:
        self.assertEqual(_clean_str("  Moonlit Driver  "), "Moonlit Driver")
        self.assertIsNone(_clean_str("   "))
        self.assertEqual(_clean_list(["  deep  ", "", None, "driving"]), ["deep", "driving"])
        self.assertEqual(
            _clean_dict_str({" Intro ": " 32 bars ", "": "skip", "Drop": ""}),
            {"Intro": "32 bars"},
        )

    def test_normalize_track_context_applies_defaults_and_coercions(self) -> None:
        context = normalize_track_context(
            {
                "workflow_mode": "invalid_mode",
                "current_stage": "unsupported",
                "bpm": "124.7",
            }
        )

        self.assertEqual(context.track_id, "default_track")
        self.assertEqual(context.workflow_mode, "general")
        self.assertIsNone(context.current_stage)
        self.assertEqual(context.bpm, 124)

    def test_normalize_track_context_rejects_invalid_bpm(self) -> None:
        context = normalize_track_context({"track_id": "moonlit_driver", "bpm": "fast"})
        self.assertIsNone(context.bpm)


class TrackContextPersistenceTests(unittest.TestCase):
    def test_save_load_load_or_create_update_and_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackContextService(make_config(root))

            created = service.load_or_create("moonlit_driver")
            self.assertEqual(created.track_id, "moonlit_driver")
            self.assertEqual(created.workflow_mode, "general")

            updated = service.update_fields(
                "moonlit_driver",
                {
                    "track_name": "Moonlit Driver",
                    "genre": "progressive house",
                    "bpm": "124",
                    "vibe": ["driving", "emotional"],
                    "sections": {"Intro": "Establish groove"},
                },
            )
            self.assertEqual(updated.track_name, "Moonlit Driver")
            self.assertEqual(updated.bpm, 124)
            self.assertEqual(updated.sections, {"Intro": "Establish groove"})

            loaded = service.load("moonlit_driver")
            self.assertEqual(loaded.track_name, "Moonlit Driver")
            self.assertEqual(loaded.vibe, ["driving", "emotional"])

            empty_path = service.yaml_directory / "empty_track.yaml"
            empty_path.parent.mkdir(parents=True, exist_ok=True)
            empty_path.write_text("", encoding="utf-8")
            empty_loaded = service.load("empty_track")
            self.assertEqual(empty_loaded.track_id, "empty_track")
            self.assertEqual(empty_loaded.workflow_mode, "general")


class TrackContextPromptTests(unittest.TestCase):
    def test_yaml_track_context_injects_into_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Help me finish this track.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                workflow_input=WorkflowInput(track_context_path="Projects/Legacy/track_context.md"),
                track_id="moonlit_driver",
                use_track_context=True,
                track_context=TrackContext(
                    track_id="moonlit_driver",
                    track_name="Moonlit Driver",
                    workflow_mode="arrangement",
                    current_stage="arrangement",
                    notes=["Shorten the intro."],
                ),
            )

            self.assertIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)
            self.assertIn("Track Id: moonlit_driver", payload.system_prompt)
            self.assertIn("Track Name: Moonlit Driver", payload.system_prompt)
            self.assertIn("Shorten the intro.", payload.system_prompt)
            self.assertNotIn("Projects/Legacy", payload.system_prompt)

    def test_yaml_track_context_takes_precedence_over_legacy_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            (project_dir / "track_context.md").write_text(
                "---\ntrack_title: Legacy Markdown Title\n---\n\nLegacy body\n",
                encoding="utf-8",
            )
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Critique this track.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver"),
                track_id="moonlit_driver",
                use_track_context=True,
                track_context=TrackContext(track_id="moonlit_driver", track_name="YAML Title"),
            )

            self.assertIn("YAML Title", payload.system_prompt)
            self.assertNotIn("Legacy Markdown Title", payload.system_prompt)
            self.assertEqual(payload.system_prompt.count("BEGIN INTERNAL TRACK CONTEXT"), 1)

    def test_arrangement_chunks_are_labeled_as_reference_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "How can I improve my arrangement?",
                [
                    RetrievedChunk(
                        text="Kick stays partial and energy remains restrained.",
                        metadata={
                            "note_title": "Tripchain Arrangement",
                            "source_path": "Knowledge/Arrangement/Tripchain.md",
                            "source_type": "track_arrangement",
                            "heading_context": "S1 - Intro",
                            "arrangement_track_name": "Tripchain",
                            "arrangement_section_name": "Intro",
                        },
                    )
                ],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
            )

            self.assertIn("Type: Arrangement reference", payload.user_prompt)
            self.assertIn("Arrangement Track: Tripchain", payload.user_prompt)
            self.assertIn("Arrangement Section: Intro", payload.user_prompt)

    def test_curated_knowledge_chunks_are_labeled_as_reference_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "What does this reference note suggest?",
                [
                    RetrievedChunk(
                        text="Progressive house benefits from long-form tension arcs.",
                        metadata={
                            "note_title": "Arrangement Principles",
                            "source_path": "Knowledge/Arrangement/Principles.md",
                            "content_scope": "knowledge",
                            "content_category": "curated_knowledge",
                        },
                    )
                ],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
            )

            self.assertIn("Type: Reference evidence", payload.user_prompt)
            self.assertIn("[Ref 1] Arrangement Principles (Knowledge/Arrangement/Principles.md)", payload.user_prompt)

    def test_imported_chunks_are_labeled_as_imported_reference_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "What does this import say?",
                [
                    RetrievedChunk(
                        text="The breakdown expands with filtered hats and vocal atmosphere.",
                        metadata={
                            "note_title": "Imported Breakdown Notes",
                            "source_path": "Imports/Web Imports/Progressive House/example.md",
                            "source_kind": "imported_content",
                            "content_scope": "knowledge",
                            "content_category": "imported_knowledge",
                        },
                    )
                ],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
            )

            self.assertIn("Type: Imported reference evidence", payload.user_prompt)
            self.assertIn("[Import 1] Imported Breakdown Notes (Imports/Web Imports/Progressive House/example.md)", payload.user_prompt)

    def test_yaml_track_context_is_not_used_without_flag_and_track_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            (project_dir / "track_context.md").write_text(
                "---\ntrack_title: Legacy Markdown Title\n---\n",
                encoding="utf-8",
            )
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Critique this track.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver"),
                track_context=TrackContext(track_id="moonlit_driver", track_name="YAML Title"),
            )

            self.assertIn("Legacy Markdown Title", payload.system_prompt)
            self.assertNotIn("YAML Title", payload.system_prompt)

    def test_prompt_keeps_track_context_above_retrieval_context_and_question_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "How should I arrange the drop?",
                [
                    RetrievedChunk(
                        text="Use contrast before the drop.",
                        metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
                        distance_or_score=0.1,
                    )
                ],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                track_id="moonlit_driver",
                use_track_context=True,
                track_context=TrackContext(track_id="moonlit_driver", track_name="Moonlit Driver"),
            )

            self.assertIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)
            self.assertIn("Local note context:", payload.user_prompt)
            self.assertGreater(payload.user_prompt.index("Question: How should I arrange the drop?"), payload.user_prompt.index("Local note context:"))

    def test_yaml_formatter_skips_empty_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Help me finish this track.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                track_id="moonlit_driver",
                use_track_context=True,
                track_context=TrackContext(track_id="moonlit_driver"),
            )

            self.assertNotIn("Genre: None", payload.system_prompt)
            self.assertNotIn("BPM: None", payload.system_prompt)

    def test_track_critique_workflow_mode_adds_structured_critique_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Please critique this track.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                track_id="moonlit_driver",
                use_track_context=True,
                track_context=TrackContext(
                    track_id="moonlit_driver",
                    workflow_mode="track_critique",
                    current_section="first drop",
                    known_issues=["drop feels flat"],
                    goals=["stronger groove"],
                ),
            )

            self.assertIn("professional electronic music producer giving structured track critique", payload.system_prompt)
            self.assertIn("1. What is working", payload.system_prompt)
            self.assertIn("5. Optional production experiments", payload.system_prompt)
            self.assertIn("known issues, goals, and current section", payload.system_prompt)
            self.assertIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)

    def test_non_critique_workflow_mode_does_not_add_structured_critique_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Help with this arrangement.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                track_id="moonlit_driver",
                use_track_context=True,
                track_context=TrackContext(
                    track_id="moonlit_driver",
                    workflow_mode="arrangement",
                ),
            )

            self.assertNotIn("professional electronic music producer giving structured track critique", payload.system_prompt)
            self.assertNotIn("1. What is working", payload.system_prompt)


class TrackContextQueryAndSaveTests(unittest.TestCase):
    def test_query_request_loads_or_creates_yaml_track_context(self) -> None:
        tracking: dict[str, object] = {"last_prompt": None, "retrieval_query": None}

        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload):
                tracking["last_prompt"] = prompt_payload
                return "Grounded answer [Local 1]."

        class StubRetriever:
            def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
                pass

            def retrieve(self, query: str, filters=None, options=None):
                tracking["retrieval_query"] = query
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
                    question="Help me finish this track.",
                    track_id="moonlit_driver",
                    use_track_context=True,
                )
            )

            self.assertIsNotNone(response.track_context)
            self.assertEqual(response.track_context.track_id, "moonlit_driver")
            self.assertTrue(service.track_context_service.exists("moonlit_driver"))
            self.assertEqual(response.debug.rewritten_query, tracking["retrieval_query"])
            self.assertEqual(response.answer_result.answer.split()[0], "Grounded")
            self.assertIn("Track Id: moonlit_driver", tracking["last_prompt"].system_prompt)

    def test_query_service_uses_rewritten_query_for_retrieval_but_keeps_original_question(self) -> None:
        tracking: dict[str, object] = {"retrieval_query": None, "last_prompt": None}

        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload):
                tracking["last_prompt"] = prompt_payload
                return "Grounded answer [Local 1]."

        class StubRetriever:
            def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
                pass

            def retrieve(self, query: str, filters=None, options=None):
                tracking["retrieval_query"] = query
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
                    question="help with the bassline",
                    track_id="moonlit_driver",
                    use_track_context=True,
                    track_context=TrackContext(
                        track_id="moonlit_driver",
                        genre="progressive house",
                        current_section="first drop",
                        known_issues=["drop feels flat"],
                        bpm=126,
                        key="A minor",
                    ),
                )
            )

            self.assertEqual(tracking["last_prompt"].user_prompt.count("Question: help with the bassline"), 1)
            self.assertEqual(
                tracking["retrieval_query"],
                "help with the bassline progressive house first drop drop feels flat 126 A minor",
            )
            self.assertEqual(response.debug.rewritten_query, tracking["retrieval_query"])

    def test_saved_outputs_include_track_context_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "output"
            output_path.mkdir()
            track_context = TrackContext(
                track_id="moonlit_driver",
                track_name="Moonlit Driver",
                workflow_mode="arrangement",
                goals=["Finish the first drop"],
            )

            destination = save_answer(
                output_path,
                "How should I finish this arrangement?",
                AnswerResult(
                    answer="Grounded answer [Local 1].",
                    sources=["[Local 1] Track Note (track.md)"],
                    retrieved_chunks=[],
                ),
                track_context=track_context,
            )

            contents = destination.read_text(encoding="utf-8")
            self.assertIn("## Track Context", contents)
            self.assertIn("Track ID: moonlit_driver", contents)
            self.assertIn("Goals: Finish the first drop", contents)
            self.assertIn("## Summary", contents)

    def test_track_context_summary_formatter_is_empty_when_missing(self) -> None:
        self.assertEqual(format_track_context_summary(None), "")
