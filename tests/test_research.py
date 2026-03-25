"""Tests for the explicit research workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import QueryDebugInfo, QueryResponse, ResearchRequest, TrackContext, WorkflowInput
from services.research_service import ResearchService
from utils import AnswerResult, RetrievedChunk
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


class ResearchWorkflowTests(unittest.TestCase):
    def test_subquestion_generation_flow(self) -> None:
        service, tracking = make_research_service()

        response = service.research(ResearchRequest(goal="Compare my notes on AI agents with recent external context"))

        self.assertGreaterEqual(len(response.subquestions), 2)
        self.assertEqual(response.subquestions[0], "What do my notes say about AI agents?")
        self.assertEqual(tracking["plan_calls"], 1)

    def test_multi_step_evidence_gathering_flow(self) -> None:
        service, tracking = make_research_service()

        response = service.research(ResearchRequest(goal="Compare my notes on AI agents with recent external context"))

        self.assertEqual(len(response.steps), 2)
        self.assertEqual(tracking["query_calls"], 2)
        self.assertIn("Agents", response.steps[0].response.sources[0])

    def test_final_synthesis_structure(self) -> None:
        service, tracking = make_research_service(
            synthesis_text="[Local 1] My notes say agents use tools. [Web 1] adds recent context.\n\n[Inference] Together this suggests agent workflows are becoming more tool-centric."
        )

        response = service.research(ResearchRequest(goal="Compare my notes on AI agents with recent external context"))

        self.assertIn("[Inference]", response.answer)
        self.assertIn("[Local 1]", response.answer)
        self.assertEqual(tracking["synthesis_calls"], 1)

    def test_source_labels_are_preserved_in_research_output(self) -> None:
        service, _ = make_research_service(
            synthesis_text="[Local 1] says agents use tools. [Saved 1] captures a prior synthesis. [Web 1] adds recent context."
        )

        response = service.research(ResearchRequest(goal="Compare my notes on AI agents with recent external context"))

        self.assertTrue(any(source.startswith("[Local") for source in response.sources))
        self.assertTrue(any(source.startswith("[Saved") for source in response.sources))
        self.assertTrue(any(source.startswith("[Web") for source in response.sources))

    def test_research_mode_handles_weak_evidence_gracefully(self) -> None:
        service, tracking = make_research_service(all_weak=True)

        response = service.research(
            ResearchRequest(
                goal="What do my notes say about a topic with little support?",
                answer_mode="strict",
            )
        )

        self.assertIn("Insufficient evidence", response.answer)
        self.assertEqual(tracking["synthesis_calls"], 0)
        self.assertTrue(any("Strict research mode limited" in warning for warning in response.warnings))

    def test_research_save_uses_research_sessions_folder(self) -> None:
        service, _ = make_research_service()
        response = service.research(ResearchRequest(goal="Compare my notes on AI agents with recent external context"))

        saved = service.save(response.goal, response.answer_result, existing_response=response)

        self.assertIsNotNone(saved.saved_path)
        self.assertIn("research_sessions", str(saved.saved_path))
        contents = saved.saved_path.read_text(encoding="utf-8")
        self.assertIn('source_type: "research_session"', contents)
        self.assertIn('status: "research"', contents)
        self.assertIn("indexed: false", contents)
        self.assertIn('workflow_type: "research_session"', contents)

    def test_research_prompt_receives_music_workflow_context(self) -> None:
        service, tracking = make_research_service()

        service.research(
            ResearchRequest(
                goal="Compare melodic techno arrangement conventions",
                workflow_input=WorkflowInput(genre="melodic techno", track_length="6:00"),
            )
        )

        self.assertIn("Domain profile: electronic_music", tracking["last_plan_prompt"])
        self.assertIn("Genre: melodic techno", tracking["last_plan_prompt"])

    def test_research_uses_chat_model_override_for_planning_and_steps(self) -> None:
        service, tracking = make_research_service()

        response = service.research(
            ResearchRequest(
                goal="Compare my notes on AI agents with recent external context",
                chat_model_override="deepseek-r1",
            )
        )

        self.assertEqual(tracking["last_model"], "deepseek-r1")
        self.assertEqual(tracking["last_query_model_override"], "deepseek-r1")
        self.assertEqual(response.active_chat_model, "deepseek-r1")

    def test_research_reuses_same_track_context_instance_for_steps_and_save(self) -> None:
        service, tracking = make_research_service()

        response = service.research(
            ResearchRequest(
                goal="Compare my notes on AI agents with recent external context",
                track_id="moonlit_driver",
                use_track_context=True,
            )
        )

        self.assertIsNotNone(response.track_context)
        self.assertIs(tracking["last_query_track_context"], response.track_context)


def make_research_service(
    *,
    synthesis_text: str = "[Local 1] My notes say agents use tools. [Web 1] adds recent context.",
    all_weak: bool = False,
) -> tuple[ResearchService, dict[str, int]]:
    tracking = {
        "plan_calls": 0,
        "synthesis_calls": 0,
        "query_calls": 0,
        "last_plan_prompt": "",
        "last_model": "",
        "last_query_model_override": "",
        "last_query_track_context": None,
    }

    class StubChatClient:
        def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
            tracking["last_model"] = model_override or config.ollama_chat_model

        def answer_with_prompt(self, prompt_payload):
            if "Generate" in prompt_payload.user_prompt and "subquestions" in prompt_payload.user_prompt:
                tracking["plan_calls"] += 1
                tracking["last_plan_prompt"] = prompt_payload.user_prompt
                return (
                    "What do my notes say about AI agents?\n"
                    "What recent external context is relevant to AI agents?"
                )
            tracking["synthesis_calls"] += 1
            return synthesis_text

    class StubQueryService:
        def __init__(self, config: AppConfig) -> None:
            self.track_context_service = self

        def load_or_create(self, track_id: str) -> TrackContext:
            return TrackContext(track_id=track_id)

        def ask(self, request):
            tracking["query_calls"] += 1
            tracking["last_query_model_override"] = request.chat_model_override or ""
            tracking["last_query_track_context"] = request.track_context
            weak_distance = 0.95 if all_weak else 0.15
            if "external context" in request.question.lower():
                return QueryResponse(
                    answer_result=AnswerResult(
                        answer="Recent external context adds examples of agent tool use [Web 1].",
                        sources=["[Web 1] Agents Update (https://example.com)"],
                        retrieved_chunks=[
                            RetrievedChunk(
                                text="Saved synthesis about agents.",
                                metadata={
                                    "note_title": "Research Answer",
                                    "source_path": "research_answers/agents-answer.md",
                                    "source_kind": "saved_answer",
                                },
                                distance_or_score=weak_distance,
                            )
                        ],
                    ),
                    web_results=[
                        WebSearchResult(
                            title="Agents Update",
                            url="https://example.com",
                            snippet="Recent update on agent tool use",
                        )
                    ],
                    debug=QueryDebugInfo(local_retrieval_weak=all_weak),
                    warnings=["External context was limited."] if all_weak else [],
                )

            return QueryResponse(
                answer_result=AnswerResult(
                    answer="My notes say AI agents use tools and retrieval [Local 1].",
                    sources=[
                        "[Local 1] Agents (agents.md)",
                        "[Saved 1] Research Answer (research_answers/agents-answer.md)",
                    ],
                    retrieved_chunks=[
                        RetrievedChunk(
                            text="Agents use tools and retrieval.",
                            metadata={
                                "note_title": "Agents",
                                "source_path": "agents.md",
                                "source_kind": "primary_note",
                            },
                            distance_or_score=weak_distance,
                        ),
                        RetrievedChunk(
                            text="Saved answer about agents.",
                            metadata={
                                "note_title": "Research Answer",
                                "source_path": "research_answers/agents-answer.md",
                                "source_kind": "saved_answer",
                            },
                            distance_or_score=weak_distance,
                        ),
                    ],
                ),
                debug=QueryDebugInfo(local_retrieval_weak=all_weak),
                warnings=["Local evidence was weak."] if all_weak else [],
            )

    root = Path(tempfile.mkdtemp())
    (root / "vault").mkdir()
    (root / "output").mkdir()
    config = make_config(root)
    service = ResearchService(
        config,
        query_service_cls=StubQueryService,
        chat_client_cls=StubChatClient,
    )
    return service, tracking
