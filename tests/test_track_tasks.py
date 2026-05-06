"""Tests for persisted per-track tasks tied to canonical YAML Track Context."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

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
    def test_load_tasks_returns_empty_when_task_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            self.assertEqual(service.load_tasks("moonlit_driver"), [])

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

    def test_task_path_sanitizes_free_text_track_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            service.add_task("../Warehouse Hypnosis/../../bad tune", text="Check the groove")

            saved_files = list(service.task_directory.glob("*.tasks.yaml"))
            self.assertEqual(len(saved_files), 1)
            self.assertEqual(saved_files[0].parent, service.task_directory)
            self.assertRegex(saved_files[0].name, r"^Warehouse_Hypnosis_bad_tune_[0-9a-f]{8}\.tasks\.yaml$")
            self.assertEqual(len(service.load_tasks("../Warehouse Hypnosis/../../bad tune")), 1)

    def test_task_path_distinguishes_sanitized_name_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            service.add_task("a/b", text="First")
            service.add_task("a b", text="Second")

            saved_names = sorted(path.name for path in service.task_directory.glob("*.tasks.yaml"))
            self.assertEqual(len(saved_names), 2)
            self.assertTrue(all(name.startswith("a_b_") for name in saved_names))
            self.assertNotEqual(saved_names[0], saved_names[1])
            self.assertEqual(service.load_tasks("a/b")[0].text, "First")
            self.assertEqual(service.load_tasks("a b")[0].text, "Second")

    def test_load_tasks_reads_existing_legacy_flat_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))
            legacy_path = service.task_directory / "Warehouse Hypnosis.tasks.yaml"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text(
                yaml.safe_dump(
                    {
                        "track_id": "Warehouse Hypnosis",
                        "schema_version": "track_tasks_v1",
                        "tasks": [
                            {
                                "id": "task-1",
                                "text": "Check the old memory",
                                "status": "open",
                                "priority": "medium",
                                "linked_section": "",
                                "created_from": "user",
                                "created_at": "2026-01-01T00:00:00",
                                "completed_at": None,
                                "notes": "",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            loaded = service.load_tasks("Warehouse Hypnosis")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].text, "Check the old memory")
            self.assertFalse(any("_" in path.name for path in service.task_directory.glob("*.tasks.yaml")))

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
            self.assertEqual(completed.status, "done")
            self.assertIsNotNone(completed.completed_at)
            reloaded_after_complete = service.load_tasks("moonlit_driver")
            self.assertEqual(reloaded_after_complete[0].status, "done")
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

    def test_deferring_task_clears_completion_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            created = service.add_task(
                "moonlit_driver",
                text="Try alternate break FX tail",
                priority="medium",
                linked_section="break",
                notes="Keep it subtle",
            )
            completed = service.complete_task("moonlit_driver", created.id, completed=True)
            assert completed is not None

            deferred = service.update_task(
                "moonlit_driver",
                created.id,
                {"status": "deferred"},
            )

            self.assertIsNotNone(deferred)
            assert deferred is not None
            self.assertEqual(deferred.status, "deferred")
            self.assertIsNone(deferred.completed_at)
            self.assertEqual(deferred.id, created.id)
            self.assertEqual(deferred.text, created.text)
            self.assertEqual(deferred.priority, created.priority)
            self.assertEqual(deferred.linked_section, created.linked_section)
            self.assertEqual(deferred.notes, created.notes)

    def test_reopening_done_task_clears_completion_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            created = service.add_task(
                "moonlit_driver",
                text="Fix the first drop transition",
                priority="high",
                linked_section="drop",
                notes="Focus on the snare lift into the hit",
            )
            completed = service.complete_task("moonlit_driver", created.id, completed=True)
            assert completed is not None

            reopened = service.complete_task("moonlit_driver", created.id, completed=False)

            self.assertIsNotNone(reopened)
            assert reopened is not None
            self.assertEqual(reopened.status, "open")
            self.assertIsNone(reopened.completed_at)
            self.assertEqual(reopened.id, created.id)
            self.assertEqual(reopened.text, created.text)
            self.assertEqual(reopened.priority, created.priority)
            self.assertEqual(reopened.linked_section, created.linked_section)
            self.assertEqual(reopened.notes, created.notes)

    def test_load_normalizes_legacy_completed_status_and_save_writes_canonical_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))
            task_path = service.task_path("moonlit_driver")
            task_path.parent.mkdir(parents=True, exist_ok=True)
            task_path.write_text(
                yaml.safe_dump(
                    {
                        "track_id": "moonlit_driver",
                        "schema_version": "track_tasks_v1",
                        "tasks": [
                            {
                                "id": "legacy-1",
                                "text": "Legacy completed task",
                                "status": "completed",
                                "priority": "medium",
                                "linked_section": "drop",
                                "created_from": "user",
                                "created_at": "2026-03-27T10:00:00Z",
                                "completed_at": "2026-03-27T11:00:00Z",
                                "notes": "",
                            },
                            {
                                "id": "legacy-2",
                                "text": "Bad status task",
                                "status": "mystery",
                                "priority": "medium",
                                "linked_section": "",
                                "created_from": "user",
                                "created_at": "2026-03-27T10:00:00Z",
                                "completed_at": "2026-03-27T11:00:00Z",
                                "notes": "",
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            loaded = service.load_tasks("moonlit_driver")

            self.assertEqual(loaded[0].status, "done")
            self.assertEqual(loaded[0].completed_at, "2026-03-27T11:00:00Z")
            self.assertEqual(loaded[1].status, "open")
            self.assertIsNone(loaded[1].completed_at)

            service.save_tasks("moonlit_driver", loaded)
            saved = yaml.safe_load(task_path.read_text(encoding="utf-8"))

            self.assertEqual(saved["tasks"][0]["status"], "done")
            self.assertEqual(saved["tasks"][1]["status"], "open")
            self.assertIsNone(saved["tasks"][1]["completed_at"])

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

    def test_tasks_are_isolated_per_track_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = TrackTaskService(make_config(root))

            service.add_task("moonlit_driver", text="Trim intro by 8 bars")
            service.add_task("nightglass", text="Tighten break tension")

            moonlit_tasks = service.load_tasks("moonlit_driver")
            nightglass_tasks = service.load_tasks("nightglass")

            self.assertEqual([task.text for task in moonlit_tasks], ["Trim intro by 8 bars"])
            self.assertEqual([task.text for task in nightglass_tasks], ["Tighten break tension"])

    def test_task_persistence_does_not_mutate_canonical_track_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            track_context_service = TrackContextService(config)
            task_service = TrackTaskService(config)
            original_context = TrackContext(
                track_id="moonlit_driver",
                track_name="Moonlit Driver",
                genre="progressive house",
                current_problem="Drop loses momentum",
            )
            track_context_service.save_canonical_track_context(original_context)

            task_service.add_task(
                "moonlit_driver",
                text="Increase drop contrast with pre-drop subtraction",
                linked_section="drop",
            )
            task_service.complete_task(
                "moonlit_driver",
                task_service.load_tasks("moonlit_driver")[0].id,
                completed=True,
            )

            reloaded_context = track_context_service.load_canonical_track_context("moonlit_driver")

            self.assertEqual(reloaded_context.track_id, original_context.track_id)
            self.assertEqual(reloaded_context.track_name, original_context.track_name)
            self.assertEqual(reloaded_context.genre, original_context.genre)
            self.assertEqual(reloaded_context.current_problem, original_context.current_problem)


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

    def test_query_service_passes_only_open_tasks_into_retrieval_and_reports_debug_counts(self) -> None:
        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubChatClient:
            def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
                self.model = model_override or config.ollama_chat_model

            def answer_with_prompt(self, prompt_payload):
                return prompt_payload.system_prompt

        class StubRetriever:
            last_current_tasks = None

            def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
                pass

            def retrieve(self, query: str, filters=None, options=None, retrieval_scope=None, **kwargs):
                StubRetriever.last_current_tasks = list(kwargs.get("current_tasks", []))
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
            config.framework_debug = True
            track_context_service = TrackContextService(config)
            track_context_service.save_canonical_track_context(
                TrackContext(track_id="moonlit_driver", track_name="Moonlit Driver")
            )
            track_task_service = TrackTaskService(config)
            open_task = track_task_service.add_task(
                "moonlit_driver",
                text="Increase drop contrast with pre-drop subtraction",
                priority="high",
                linked_section="drop",
            )
            done_task = track_task_service.add_task(
                "moonlit_driver",
                text="Test alternate clap layer in break",
                linked_section="break",
            )
            deferred_task = track_task_service.add_task(
                "moonlit_driver",
                text="Revisit intro riser sweep",
                linked_section="intro",
            )
            track_task_service.complete_task("moonlit_driver", done_task.id, completed=True)
            track_task_service.update_task(
                "moonlit_driver",
                deferred_task.id,
                {"status": "deferred"},
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
                    question="What should I do next on this track?",
                    track_id="moonlit_driver",
                    use_track_context=True,
                    collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                )
            )

            self.assertIsNotNone(StubRetriever.last_current_tasks)
            retrieval_tasks = StubRetriever.last_current_tasks
            assert retrieval_tasks is not None
            self.assertEqual([task.id for task in retrieval_tasks], [open_task.id])
            self.assertEqual(response.debug.loaded_task_count, 3)
            self.assertEqual(response.debug.open_task_count, 1)
            self.assertEqual(
                response.debug.active_task_summaries,
                (f"{open_task.id}:{open_task.text}",),
            )
