"""Explicit multi-step research workflow built on top of the existing query service."""

from __future__ import annotations

import re

from config import AppConfig
from llm import OllamaChatClient
from saver import save_answer
from services.music_workflow_service import MusicWorkflowService
from services.models import (
    AnswerMode,
    CollaborationWorkflow,
    DomainProfile,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchStepResult,
    RetrievalMode,
    WorkflowInput,
)
from services.prompt_service import PromptService, build_citation_sources, enforce_citation_summary
from services.query_service import QueryService
from utils import AnswerResult, RetrievedChunk, get_logger
from web_search import WebSearchResult


logger = get_logger()


class ResearchService:
    """Coordinate a visible, bounded research workflow without hidden autonomy."""

    def __init__(
        self,
        config: AppConfig,
        *,
        query_service_cls: type[QueryService] = QueryService,
        chat_client_cls: type[OllamaChatClient] = OllamaChatClient,
        prompt_service_cls: type[PromptService] = PromptService,
    ) -> None:
        self.config = config
        self.query_service_cls = query_service_cls
        self.chat_client_cls = chat_client_cls
        self.prompt_service_cls = prompt_service_cls
        self.music_workflow_service = MusicWorkflowService(config)

    def research(self, request: ResearchRequest) -> ResearchResponse:
        """Run a bounded multi-step research workflow and return structured results."""
        query_service = self.query_service_cls(self.config)
        prompt_service = self.prompt_service_cls(self.config)
        chat_client = _build_chat_client(
            self.chat_client_cls,
            self.config,
            model_override=request.chat_model_override,
        )
        workflow_plan = self.music_workflow_service.build_research_plan(request)
        track_context = None
        if request.use_track_context and request.track_id:
            track_context = query_service.track_context_service.load_or_create(request.track_id)

        subquestions, planning_notes = self._generate_subquestions(
            workflow_plan.prompt_text,
            answer_mode=request.answer_mode,
            max_subquestions=request.max_subquestions,
            chat_client=chat_client,
            prompt_service=prompt_service,
            domain_profile=request.domain_profile,
            workflow_input=request.workflow_input,
        )

        steps: list[ResearchStepResult] = []
        for subquestion in subquestions:
            step_response = query_service.ask(
                QueryRequest(
                    question=subquestion,
                    filters=request.filters,
                    options=request.options,
                    retrieval_scope=request.retrieval_scope,
                    retrieval_mode=request.retrieval_mode,
                    answer_mode=request.answer_mode,
                    domain_profile=request.domain_profile,
                    collaboration_workflow=request.collaboration_workflow,
                    workflow_input=request.workflow_input,
                    track_id=request.track_id,
                    use_track_context=request.use_track_context,
                    track_context=track_context,
                    chat_model_override=request.chat_model_override,
                )
            )
            steps.append(ResearchStepResult(subquestion=subquestion, response=step_response))

        final_answer_result, final_warnings = self._synthesize_research_answer(
            request.goal,
            steps,
            answer_mode=request.answer_mode,
            retrieval_mode=request.retrieval_mode,
            chat_client=chat_client,
            prompt_service=prompt_service,
            domain_profile=request.domain_profile,
            workflow_input=request.workflow_input,
        )

        warnings = list(dict.fromkeys(planning_notes + final_warnings + _collect_step_warnings(steps)))
        saved_path = None
        if request.auto_save or self.config.auto_save_answer:
            saved_path = save_answer(
                workflow_plan.save_path,
                request.goal,
                final_answer_result,
                title_override=request.save_title,
                source_type="research_session",
                status="research",
                indexed=False,
                domain_profile=request.domain_profile.value,
                workflow_type=request.collaboration_workflow.value,
                workflow_input=request.workflow_input.as_dict(),
                track_context=track_context,
            )
            logger.info("Saved research answer to %s", saved_path)

        return ResearchResponse(
            goal=request.goal,
            subquestions=subquestions,
            steps=steps,
            answer_result=final_answer_result,
            warnings=warnings,
            saved_path=saved_path,
            planning_notes=planning_notes,
            active_chat_model=getattr(chat_client, "model", self.config.ollama_chat_model),
            domain_profile=request.domain_profile,
            collaboration_workflow=request.collaboration_workflow,
            workflow_input=request.workflow_input,
            track_context=track_context,
        )

    def save(
        self,
        goal: str,
        answer_result: AnswerResult,
        *,
        title_override: str | None = None,
        existing_response: ResearchResponse | None = None,
    ) -> ResearchResponse:
        """Persist an existing research answer result and preserve prior workflow state."""
        saved_path = save_answer(
            self.music_workflow_service.default_save_path(
                existing_response.collaboration_workflow
                if existing_response is not None
                else CollaborationWorkflow.RESEARCH_SESSION
            ),
            goal,
            answer_result,
            title_override=title_override,
            source_type="research_session",
            status="research",
            indexed=False,
            domain_profile=(
                existing_response.domain_profile.value if existing_response is not None else None
            ),
            workflow_type=(
                existing_response.collaboration_workflow.value if existing_response is not None else None
            ),
            workflow_input=(
                existing_response.workflow_input.as_dict() if existing_response is not None else None
            ),
            track_context=(
                existing_response.track_context if existing_response is not None else None
            ),
        )
        logger.info("Saved research answer to %s", saved_path)
        if existing_response is not None:
            return existing_response.with_saved_path(saved_path)
        return ResearchResponse(
            goal=goal,
            subquestions=[],
            steps=[],
            answer_result=answer_result,
            saved_path=saved_path,
            active_chat_model=(
                existing_response.active_chat_model if existing_response is not None else ""
            ),
            domain_profile=(
                existing_response.domain_profile
                if existing_response is not None
                else DomainProfile.ELECTRONIC_MUSIC
            ),
            collaboration_workflow=(
                existing_response.collaboration_workflow
                if existing_response is not None
                else CollaborationWorkflow.RESEARCH_SESSION
            ),
            workflow_input=(
                existing_response.workflow_input
                if existing_response is not None
                else WorkflowInput()
            ),
            track_context=(
                existing_response.track_context
                if existing_response is not None
                else None
            ),
        )
    def _generate_subquestions(
        self,
        goal: str,
        *,
        answer_mode: AnswerMode,
        max_subquestions: int,
        chat_client: OllamaChatClient,
        prompt_service: PromptService,
        domain_profile: DomainProfile,
        workflow_input: WorkflowInput,
    ) -> tuple[list[str], list[str]]:
        planning_notes: list[str] = []
        payload = prompt_service.build_research_plan_payload(
            goal,
            answer_mode=answer_mode,
            max_subquestions=max_subquestions,
            domain_profile=domain_profile,
            workflow_input=workflow_input,
        )
        try:
            raw_plan = chat_client.answer_with_prompt(payload)
            subquestions = _parse_subquestions(raw_plan, max_subquestions=max_subquestions)
        except Exception as exc:
            planning_notes.append(f"Research planning fallback was used: {exc}")
            subquestions = []

        if not subquestions:
            planning_notes.append("Research subquestions were generated with a deterministic fallback.")
            subquestions = _fallback_subquestions(goal, max_subquestions=max_subquestions)

        return subquestions[:max_subquestions], planning_notes

    def _synthesize_research_answer(
        self,
        goal: str,
        steps: list[ResearchStepResult],
        *,
        answer_mode: AnswerMode,
        retrieval_mode: RetrievalMode,
        chat_client: OllamaChatClient,
        prompt_service: PromptService,
        domain_profile: DomainProfile,
        workflow_input: WorkflowInput,
    ) -> tuple[AnswerResult, list[str]]:
        combined_chunks = _dedupe_chunks([chunk for step in steps for chunk in step.response.retrieved_chunks])
        combined_web = _dedupe_web_results([result for step in steps for result in step.response.web_results])
        citation_sources, citation_labels = build_citation_sources(combined_chunks, combined_web)

        if not citation_sources:
            answer = (
                "Insufficient evidence was gathered across the research steps to produce a grounded research answer."
            )
            return (
                AnswerResult(
                    answer=answer,
                    sources=[],
                    retrieved_chunks=combined_chunks,
                ),
                ["Research mode could not gather enough evidence across the planned subquestions."],
            )

        if answer_mode == AnswerMode.STRICT and all(_response_is_weak(step.response) for step in steps):
            answer = (
                "Insufficient evidence was gathered across the research steps to answer this in Strict mode."
            )
            answer = enforce_citation_summary(answer, tuple(citation_labels), answer_mode)
            return (
                AnswerResult(
                    answer=answer,
                    sources=citation_sources,
                    retrieved_chunks=combined_chunks,
                ),
                ["Strict research mode limited the final synthesis because the step evidence remained weak."],
            )

        step_findings = [
            (step.subquestion, step.response.answer, step.response.sources, step.response.warnings)
            for step in steps
        ]
        payload = prompt_service.build_research_synthesis_payload(
            goal,
            step_findings,
            answer_mode=answer_mode,
            retrieval_mode=retrieval_mode,
            citation_sources=citation_sources,
            domain_profile=domain_profile,
            workflow_input=workflow_input,
        )
        answer = chat_client.answer_with_prompt(payload)
        answer = enforce_citation_summary(answer, tuple(citation_labels), answer_mode)
        return (
            AnswerResult(
                answer=answer,
                sources=citation_sources,
                retrieved_chunks=combined_chunks,
            ),
            [],
        )


def _build_chat_client(
    chat_client_cls: type[OllamaChatClient],
    config: AppConfig,
    *,
    model_override: str | None,
) -> OllamaChatClient:
    """Instantiate a chat client with optional model override and test-stub compatibility."""
    if model_override:
        try:
            client = chat_client_cls(config, model_override=model_override)
            setattr(client, "model", model_override)
            return client
        except TypeError:
            client = chat_client_cls(config)
            setattr(client, "model", model_override)
            return client
    return chat_client_cls(config)


def _parse_subquestions(raw_plan: str, *, max_subquestions: int) -> list[str]:
    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in raw_plan.splitlines()
    ]
    candidates = [line for line in lines if line]
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in seen or len(candidate) < 8:
            continue
        seen.add(normalized)
        deduped.append(candidate.rstrip("?") + "?")
        if len(deduped) >= max_subquestions:
            break
    return deduped


def _fallback_subquestions(goal: str, *, max_subquestions: int) -> list[str]:
    templates = [
        f"What do my notes most directly say about: {goal}?",
        f"What evidence in my notes supports or qualifies: {goal}?",
        f"What external context is most relevant to: {goal}?",
    ]
    return templates[:max_subquestions]


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    deduped: list[RetrievedChunk] = []
    seen: set[tuple[object, object, object]] = set()
    for chunk in chunks:
        key = (
            chunk.metadata.get("source_path"),
            chunk.metadata.get("chunk_index"),
            chunk.metadata.get("note_title"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _dedupe_web_results(results: list[WebSearchResult]) -> list[WebSearchResult]:
    deduped: list[WebSearchResult] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        key = (result.title, result.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _collect_step_warnings(steps: list[ResearchStepResult]) -> list[str]:
    warnings: list[str] = []
    for step in steps:
        for warning in step.response.warnings:
            if warning not in warnings:
                warnings.append(warning)
    return warnings


def _response_is_weak(response: QueryResponse) -> bool:
    return response.debug.local_retrieval_weak or not response.sources
