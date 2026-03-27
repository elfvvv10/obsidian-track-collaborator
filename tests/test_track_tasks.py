"""Tests for persisted per-track tasks tied to canonical YAML Track Context."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import CollaborationWorkflow, QueryRequest, TrackContext
from services.query_service import QueryService
from services.track_context_service import TrackContextService
from services.track_task_service import TrackTaskService
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


class TrackTaskServiceTests(unittest.TestCase):
    def test_save_and_load_tasks_by_track_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            created = service.add_task(
                "moonlit_driver",
                text="Shorten intro by 8 bars",
                created_from="user",
                priority="high",
                linked_section="intro",
                notes="Trim before the first fill",
            )
            loaded = service.load_tasks("moonlit_driver")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].id, created.id)
            self.assertEqual(loaded[0].priority, "high")
            self.assertEqual(loaded[0].linked_section, "intro")
            self.assertEqual(loaded[0].notes, "Trim before the first fill")

    def test_complete_update_and_delete_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            created = service.add_task("moonlit_driver", text="Fix the drop transition")
            completed = service.complete_task("moonlit_driver", created.id, completed=True)
            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertEqual(completed.status, "completed")
            self.assertIsNotNone(completed.completed_at)
            reloaded_after_complete = service.load_tasks("moonlit_driver")
            self.assertEqual(reloaded_after_complete[0].status, "completed")
            self.assertEqual(reloaded_after_complete[0].completed_at, completed.completed_at)

            updated = service.update_task(
                "moonlit_driver",
                created.id,
                {
                    "text": "Fix the first drop transition",
                    "priority": "high",
                    "linked_section": "drop",
                    "notes": "Focus on the snare lift into the hit",
                },
            )
            self.assertIsNotNone(updated)
            assert updated is not None
            self.assertEqual(updated.text, "Fix the first drop transition")
            self.assertEqual(updated.priority, "high")
            self.assertEqual(updated.linked_section, "drop")
            self.assertEqual(updated.notes, "Focus on the snare lift into the hit")

            deleted = service.delete_task("moonlit_driver", created.id)
            self.assertTrue(deleted)
            self.assertEqual(service.load_tasks("moonlit_driver"), [])

    def test_load_session_tasks_preserves_prompt_relevant_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            service.add_task(
                "moonlit_driver",
                text="Tighten the drop groove",
                priority="high",
                linked_section="drop",
                notes="Check bass syncopation",
            )
            session_tasks = service.load_session_tasks("moonlit_driver")

            self.assertEqual(len(session_tasks), 1)
            self.assertEqual(session_tasks[0].priority, "high")
            self.assertEqual(session_tasks[0].linked_section, "drop")
            self.assertEqual(session_tasks[0].notes, "Check bass syncopation")


class QueryServiceTrackTaskTests(unittest.TestCase):
    def test_query_service_automatically_loads_persisted_tasks_for_active_track(self) -> None:
        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload):
                return prompt_payload.system_prompt

        class StubRetriever:
            def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
                pass

            def retrieve(self, query: str, filters=None, options=None, retrieval_scope=None, **kwargs):
                return [
                    RetrievedChunk(
                        text="Arrangement note",
                        metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
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
            config = make_config(root)
            track_context_service = TrackContextService(config)
            track_context_service.save_canonical_track_context(
                TrackContext(track_id="moonlit_driver", track_name="Moonlit Driver")
            )
            track_task_service = TrackTaskService(config)
            track_task_service.add_task(
                "moonlit_driver",
                text="Shorten the intro by 8 bars",
                priority="high",
                linked_section="intro",
                notes="Keep the first fill",
            )

            service = QueryService(
                config,
                embedding_client_cls=StubEmbeddingClient,
                chat_client_cls=StubChatClient,
                retriever_cls=StubRetriever,
                vector_store_cls=StubVectorStore,
                web_search_service_cls=StubWebSearchService,
                capture_debug_trace=False,
            )

            response = service.ask(
                QueryRequest(
                    question="What should I do next?",
                    track_id="moonlit_driver",
                    use_track_context=True,
                    collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                )
            )

            self.assertIn("BEGIN CURRENT TASKS", response.answer)
            self.assertIn("Shorten the intro by 8 bars", response.answer)
            self.assertIn("priority: high", response.answer)
            self.assertIn("section: intro", response.answer)
