"""Lightweight regression fixtures for track-aware retrieval behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from retriever import Retriever
from services.models import (
    CollaborationWorkflow,
    DomainProfile,
    QueryRequest,
    SectionContext,
    TrackContext,
    WorkflowInput,
)
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
        framework_debug=True,
    )


def chunk(text: str, title: str, path: str, distance: float, **metadata: object) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"note_title": title, "source_path": path, "chunk_index": 0, **metadata},
        distance_or_score=distance,
    )


SCENARIO_CANDIDATES: dict[str, list[RetrievedChunk]] = {
    "stuck_track": [
        chunk(
            "breakdown re-entry urgency improves when the fill accelerates density across the last four bars",
            "Breakdown Re-entry Fixes",
            "breakdown.md",
            0.12,
            source_type="track_arrangement",
            arrangement_section_name="Breakdown",
            arrangement_genre="progressive house",
            heading_context="re-entry urgency",
            content_category="curated_knowledge",
        ),
        chunk(
            "if the first drop loses contrast after 8 bars, use bar 49 as a pivot and vary the bass motif",
            "Drop Pivot Note",
            "drop.md",
            0.22,
            source_type="track_arrangement",
            arrangement_section_name="Drop",
            arrangement_genre="progressive house",
            heading_context="bar 49 pivot",
            content_category="curated_knowledge",
        ),
        chunk(
            "when tracks feel stuck, pick one area and finish it before touching anything else",
            "Generic Finish Tracks Advice",
            "generic-finish.md",
            0.07,
            content_category="curated_knowledge",
        ),
    ],
    "drop_improvement": [
        chunk(
            "progressive house drop contrast improves when the motif evolves in the second half",
            "Progressive House Drop Dynamics",
            "genre-note.md",
            0.18,
            source_type="youtube_video",
            import_genre="progressive house",
            video_section_title="Drop contrast",
            content_category="curated_knowledge",
        ),
        chunk(
            "moonlit driver drop blueprint: keep bars 33 to 40 stable, then create a bar 49 pivot",
            "Moonlit Driver Drop Blueprint",
            "arrangement.md",
            0.22,
            source_type="track_arrangement",
            arrangement_track_name="Moonlit Driver",
            arrangement_section_name="Drop",
            arrangement_genre="progressive house",
            heading_context="drop payoff and bar 49 pivot",
            content_category="curated_knowledge",
        ),
        chunk(
            "club drops feel bigger when you add more hats and more noise every 8 bars",
            "Generic Club Notes",
            "generic-club.md",
            0.08,
            content_category="curated_knowledge",
        ),
    ],
    "arrangement_critique": [
        chunk(
            "moonlit driver drop blueprint: keep bars 33 to 40 stable, then create a bar 49 pivot",
            "Moonlit Driver Drop Blueprint",
            "arrangement.md",
            0.22,
            source_type="track_arrangement",
            arrangement_track_name="Moonlit Driver",
            arrangement_section_name="Drop",
            arrangement_genre="progressive house",
            heading_context="drop payoff and bar 49 pivot",
            content_category="curated_knowledge",
        ),
        chunk(
            "progressive house drop contrast improves when the motif evolves in the second half",
            "Progressive House Drop Dynamics",
            "genre-note.md",
            0.18,
            source_type="youtube_video",
            import_genre="progressive house",
            video_section_title="Drop contrast",
            content_category="curated_knowledge",
        ),
        chunk(
            "general club arrangement advice with weak overlap",
            "Generic Club Notes",
            "generic.md",
            0.05,
            content_category="curated_knowledge",
        ),
    ],
    "yaml_vs_legacy": [
        chunk(
            "moonlit driver drop blueprint: keep bars 33 to 40 stable, then create a bar 49 pivot",
            "Moonlit Driver Drop Blueprint",
            "arrangement.md",
            0.22,
            source_type="track_arrangement",
            arrangement_track_name="Moonlit Driver",
            arrangement_section_name="Drop",
            arrangement_genre="progressive house",
            heading_context="drop payoff and bar 49 pivot",
            content_category="curated_knowledge",
        ),
        chunk(
            "progressive house drop contrast improves when the motif evolves in the second half",
            "Progressive House Drop Dynamics",
            "genre-note.md",
            0.18,
            source_type="youtube_video",
            import_genre="progressive house",
            video_section_title="Drop contrast",
            content_category="curated_knowledge",
        ),
        chunk(
            "general club arrangement advice with weak overlap",
            "Generic Club Notes",
            "generic.md",
            0.05,
            content_category="curated_knowledge",
        ),
    ],
}


def _resolve_scenario(query: str) -> str:
    lowered = query.lower()
    if "what should i do next to get this track unstuck" in lowered:
        return "stuck_track"
    if "how can i improve the second half of the drop without making it busier" in lowered:
        return "drop_improvement"
    if "critique the arrangement of moonlit driver" in lowered:
        return "arrangement_critique"
    if "critique this track concept" in lowered:
        return "yaml_vs_legacy"
    raise KeyError(query)


class StubEmbeddingClient:
    def __init__(self, config: AppConfig) -> None:
        pass

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class StubChatClient:
    def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
        self.model = model_override or config.ollama_chat_model
        self.provider = "ollama"

    def answer_with_prompt(self, prompt_payload):
        return prompt_payload.system_prompt


class ServiceVectorStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def is_index_compatible(self) -> bool:
        return True

    def count(self) -> int:
        return 1


class CandidateVectorStore:
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self.candidates = list(candidates)

    def count(self) -> int:
        return len(self.candidates)

    def query(self, query_embedding, top_k, filters=None, **kwargs):
        return list(self.candidates)

    def get_chunks_by_note_keys(
        self,
        note_keys,
        max_chunks_per_note=1,
        retrieval_scope=None,
        excluded_note_keys=None,
    ) -> list[RetrievedChunk]:
        return []


class ScenarioRetriever:
    def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
        self.config = config
        self.embedding_client = embedding_client

    def _retriever_for(self, query: str) -> Retriever:
        return Retriever(
            self.config,
            self.embedding_client,
            CandidateVectorStore(SCENARIO_CANDIDATES[_resolve_scenario(query)]),
        )

    def retrieve(self, query: str, **kwargs):
        return self._retriever_for(query).retrieve(query, **kwargs)

    def retrieve_with_debug(self, query: str, **kwargs):
        return self._retriever_for(query).retrieve_with_debug(query, **kwargs)


class StubWebSearchService:
    def __init__(self, config: AppConfig) -> None:
        pass

    def search(self, query: str) -> list[object]:
        return []


class RetrievalRegressionTests(unittest.TestCase):
    def _build_service(self, root: Path) -> QueryService:
        config = make_config(root)
        track_context_service = TrackContextService(config)
        track_task_service = TrackTaskService(config)

        legacy_dir = root / "vault" / "Projects" / "Moonlit Driver"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        (legacy_dir / "track_context.md").write_text(
            "---\ntrack_title: Legacy Wrong Title\nprimary_genre: trance\n---\n",
            encoding="utf-8",
        )

        track_context_service.save_canonical_track_context(
            TrackContext(
                track_id="moonlit_driver",
                track_name="Moonlit Driver",
                genre="progressive house",
                current_stage="arrangement",
                current_problem="first drop loses contrast after the initial 8 bars",
                known_issues=["breakdown re-entry feels too polite"],
                goals=["finish arrangement", "improve drop contrast"],
                vibe=["euphoric", "driving"],
                reference_tracks=["Deadmau5 - Strobe"],
                sections={
                    "drop": SectionContext(
                        name="Drop",
                        bars="33-64",
                        role="main payoff",
                        energy_level="high",
                        elements=["bass motif", "lead stack", "snare lift"],
                        issues=["loses contrast after 8 bars"],
                        notes="Bar 49 pivot is the key transition point.",
                    )
                },
            )
        )
        track_task_service.add_task(
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
        track_task_service.update_task("moonlit_driver", deferred_task.id, {"status": "deferred"})

        return QueryService(
            config,
            embedding_client_cls=StubEmbeddingClient,
            chat_client_cls=StubChatClient,
            retriever_cls=ScenarioRetriever,
            vector_store_cls=ServiceVectorStore,
            web_search_service_cls=StubWebSearchService,
            capture_debug_trace=True,
        )

    def _run_scenario(
        self,
        root: Path,
        *,
        question: str,
        workflow: CollaborationWorkflow,
        section_focus: str | None = None,
        workflow_input: WorkflowInput | None = None,
    ):
        service = self._build_service(root)
        return service.ask(
            QueryRequest(
                question=question,
                track_id="moonlit_driver",
                use_track_context=True,
                collaboration_workflow=workflow,
                workflow_input=workflow_input or WorkflowInput(),
                section_focus=section_focus,
                domain_profile=DomainProfile.ELECTRONIC_MUSIC,
            )
        )

    def test_query_regressions_preserve_expected_top_rankings(self) -> None:
        scenarios = (
            (
                "What should I do next to get this track unstuck?",
                CollaborationWorkflow.ARRANGEMENT_PLANNER,
                None,
                "Drop Pivot Note",
            ),
            (
                "How can I improve the second half of the drop without making it busier?",
                CollaborationWorkflow.ARRANGEMENT_PLANNER,
                "drop",
                "Moonlit Driver Drop Blueprint",
            ),
            (
                "Critique the arrangement of Moonlit Driver, especially the drop payoff.",
                CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                "drop",
                "Moonlit Driver Drop Blueprint",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()

            for question, workflow, section_focus, expected_top_title in scenarios:
                with self.subTest(question=question):
                    response = self._run_scenario(
                        root,
                        question=question,
                        workflow=workflow,
                        section_focus=section_focus,
                    )
                    self.assertTrue(response.debug.reranking_details)
                    self.assertEqual(response.debug.reranking_details[0].note_title, expected_top_title)

    def test_task_relevance_precision_and_open_task_only_influence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()

            response = self._run_scenario(
                root,
                question="What should I do next to get this track unstuck?",
                workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
            )

            self.assertEqual(response.debug.loaded_task_count, 3)
            self.assertEqual(response.debug.open_task_count, 1)
            top_detail = response.debug.reranking_details[0]
            second_detail = response.debug.reranking_details[1]
            generic_detail = response.debug.reranking_details[2]
            self.assertEqual(top_detail.note_title, "Drop Pivot Note")
            self.assertGreater(top_detail.component_scores["task_relevance"], 0.0)
            self.assertEqual(second_detail.note_title, "Breakdown Re-entry Fixes")
            self.assertEqual(second_detail.component_scores["task_relevance"], 0.0)
            self.assertEqual(generic_detail.note_title, "Generic Finish Tracks Advice")
            self.assertEqual(generic_detail.component_scores["task_relevance"], 0.0)

    def test_active_yaml_track_context_remains_authoritative_over_legacy_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()

            response = self._run_scenario(
                root,
                question="Critique this track concept.",
                workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver/track_context.md"),
            )

            self.assertIsNotNone(response.track_context)
            assert response.track_context is not None
            self.assertEqual(response.track_context.track_name, "Moonlit Driver")
            self.assertIn("Track Name: Moonlit Driver", response.answer)
            self.assertNotIn("Legacy Wrong Title", response.answer)
