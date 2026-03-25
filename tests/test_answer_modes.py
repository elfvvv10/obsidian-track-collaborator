"""Tests for answer-mode policy behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import CollaborationWorkflow, QueryRequest, WorkflowInput
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
    tracking: dict[str, object] = {"chat_calls": 0, "last_prompt": None, "last_model": None}

    class StubEmbeddingClient:
        def __init__(self, config: AppConfig) -> None:
            pass

    class StubChatClient:
        def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
            tracking["last_model"] = model_override or config.ollama_chat_model

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

    def test_query_uses_chat_model_override_when_provided(self) -> None:
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

        response = service.ask(QueryRequest(question="agents?", chat_model_override="deepseek-r1"))

        self.assertEqual(tracking["last_model"], "deepseek-r1")
        self.assertEqual(response.debug.active_chat_model, "deepseek-r1")

    def test_music_workflow_flows_into_prompt_payload(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Breakbeat often relies on syncopated drums.",
                    metadata={"note_title": "Breakbeat Notes", "source_path": "breakbeat.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Does this idea fit breakbeat?",
                collaboration_workflow=CollaborationWorkflow.GENRE_FIT_REVIEW,
                workflow_input=WorkflowInput(genre="breakbeat", bpm="135"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.collaboration_workflow.value, "genre_fit_review")
        self.assertIn("Collaboration workflow: genre_fit_review", prompt_payload.user_prompt)
        self.assertIn("Assess likely genre or style fit", prompt_payload.user_prompt)
        self.assertIn("Genre: breakbeat", prompt_payload.user_prompt)

    def test_critique_workflow_prompt_encourages_implementation_coaching(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Track notes about weak transitions.",
                    metadata={"note_title": "Track Notes", "source_path": "track.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Critique this transition.",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(genre="progressive house"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Start with a direct answer to the user's music question", prompt_payload.user_prompt)
        self.assertIn("Do not open with framing language", prompt_payload.user_prompt)
        self.assertIn("Every meaningful suggestion must include how to do it", prompt_payload.user_prompt)
        self.assertIn("Do not stop at abstract advice", prompt_payload.user_prompt)
        self.assertIn("prioritize genre-native techniques first", prompt_payload.user_prompt)
        self.assertIn("Ignore weak, tangential, or loosely related sources", prompt_payload.user_prompt)
        self.assertIn("Treat Track Context as long-term track identity and current production state", prompt_payload.user_prompt)
        self.assertIn("analyze it section by section", prompt_payload.user_prompt)
        self.assertIn("Overall Assessment, Arrangement / Energy Flow, Genre / Style Fit, Groove / Bass / Element Evolution, Priority Issues, Recommended Next Changes", prompt_payload.user_prompt)
        self.assertIn("how to implement the change", prompt_payload.user_prompt)
        self.assertIn("first pass", prompt_payload.user_prompt)
        self.assertIn("what to listen for afterward", prompt_payload.user_prompt)
        self.assertIn("why it matters", prompt_payload.user_prompt)
        self.assertIn("provide multiple concrete, usable ideas", prompt_payload.user_prompt)

    def test_arrangement_workflow_prompt_encourages_implementation_coaching(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Arrangement notes about low energy sections.",
                    metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Plan this arrangement.",
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                workflow_input=WorkflowInput(track_length="6:00"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Start with a direct answer to the user's music question", prompt_payload.user_prompt)
        self.assertIn("Use retrieved material to support, constrain, or refine the answer after the direct answer", prompt_payload.user_prompt)
        self.assertIn("how to implement it in practical production terms", prompt_payload.user_prompt)
        self.assertIn("minimal first pass", prompt_payload.user_prompt)
        self.assertIn("what to listen for afterward", prompt_payload.user_prompt)
        self.assertIn("Every meaningful suggestion must include how to do it", prompt_payload.user_prompt)

    def test_sound_design_workflow_gains_answer_first_and_genre_grounding(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Progressive house bass design notes.",
                    metadata={"note_title": "Bass Design", "source_path": "bass.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Give me some progressive house bassline ideas.",
                collaboration_workflow=CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM,
                workflow_input=WorkflowInput(genre="progressive house"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Start with a direct answer to the user's music question", prompt_payload.user_prompt)
        self.assertIn("Every meaningful suggestion must include how to do it", prompt_payload.user_prompt)
        self.assertIn("prioritize genre-native techniques first", prompt_payload.user_prompt)
        self.assertIn("Treat cross-genre or adjacent-genre ideas as optional variations", prompt_payload.user_prompt)
        self.assertIn("provide multiple concrete, usable ideas", prompt_payload.user_prompt)
        self.assertIn("Quick Answer", prompt_payload.user_prompt)
        self.assertIn("Production Recipes", prompt_payload.user_prompt)
        self.assertIn("Groove / MIDI", prompt_payload.user_prompt)
        self.assertIn("Sound Design", prompt_payload.user_prompt)
        self.assertIn("How to Build It", prompt_payload.user_prompt)
        self.assertIn("Where to Use It", prompt_payload.user_prompt)
        self.assertIn("Do not open with phrases such as 'Based on the provided context'", prompt_payload.user_prompt)
        self.assertIn("Do not produce generic advice or filler phrases", prompt_payload.user_prompt)
        self.assertIn("Core recipes must be musically plausible for the requested genre or style", prompt_payload.user_prompt)
        self.assertIn("prioritize genre-common archetypes first", prompt_payload.user_prompt)
        self.assertIn("Weakly related or cross-genre retrieved material must not become core recommendations", prompt_payload.user_prompt)
        self.assertIn("optional inspiration or an optional variation", prompt_payload.user_prompt)
        self.assertIn("sensible starting points", prompt_payload.user_prompt)
        self.assertIn("Reject arbitrary or musically implausible ideas", prompt_payload.user_prompt)

    def test_non_target_workflow_prompt_does_not_gain_music_collaboration_contract(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Breakbeat note.",
                    metadata={"note_title": "Breakbeat", "source_path": "breakbeat.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Does this fit breakbeat?",
                collaboration_workflow=CollaborationWorkflow.GENRE_FIT_REVIEW,
                workflow_input=WorkflowInput(genre="breakbeat"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertNotIn("what to listen for afterward", prompt_payload.user_prompt)
        self.assertNotIn("how to implement the change", prompt_payload.user_prompt)
        self.assertNotIn("Start with a direct answer to the user's music question", prompt_payload.user_prompt)
        self.assertNotIn("provide multiple concrete, usable ideas", prompt_payload.user_prompt)

    def test_other_workflows_do_not_gain_sound_design_structure(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Arrangement note.",
                    metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Plan this arrangement.",
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                workflow_input=WorkflowInput(track_length="6:00"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertNotIn("Groove / MIDI", prompt_payload.user_prompt)
        self.assertNotIn("How to Build It", prompt_payload.user_prompt)
        self.assertNotIn("Production Recipes", prompt_payload.user_prompt)
        self.assertNotIn("Core recipes must be musically plausible for the requested genre or style", prompt_payload.user_prompt)
        self.assertNotIn("Weakly related or cross-genre retrieved material must not become core recommendations", prompt_payload.user_prompt)
