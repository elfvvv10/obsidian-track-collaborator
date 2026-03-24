"""Thin query service built on existing retrieval and answer modules."""

from __future__ import annotations

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from retriever import Retriever
from saver import save_answer
from services.common import ensure_index_compatible
from services.music_workflow_service import MusicWorkflowService
from services.models import (
    AnswerMode,
    CollaborationWorkflow,
    DomainProfile,
    QueryDebugInfo,
    QueryRequest,
    QueryResponse,
    RetrievalMode,
    RetrievalModeUsed,
    RetrievalScope,
    WebSearchAttemptInfo,
    WebQueryStrategy,
    WorkflowInput,
)
from services.prompt_service import PromptService, answer_uses_inference, build_citation_sources, enforce_citation_summary
from services.web_alignment_service import WebAlignmentResult, WebAlignmentService
from services.web_search_service import WebSearchService
from utils import AnswerResult, RetrievedChunk, get_logger
from vector_store import VectorStore
from web_search import WebSearchResult


logger = get_logger()


class QueryService:
    """Coordinate retrieval, answer generation, and optional save-back."""

    def __init__(
        self,
        config: AppConfig,
        *,
        embedding_client_cls: type[OllamaEmbeddingClient] = OllamaEmbeddingClient,
        chat_client_cls: type[OllamaChatClient] = OllamaChatClient,
        retriever_cls: type[Retriever] = Retriever,
        vector_store_cls: type[VectorStore] = VectorStore,
        web_search_service_cls: type[WebSearchService] = WebSearchService,
        prompt_service_cls: type[PromptService] = PromptService,
        web_alignment_service_cls: type[WebAlignmentService] = WebAlignmentService,
        capture_debug_trace: bool = True,
    ) -> None:
        self.config = config
        self.embedding_client_cls = embedding_client_cls
        self.chat_client_cls = chat_client_cls
        self.retriever_cls = retriever_cls
        self.vector_store_cls = vector_store_cls
        self.web_search_service_cls = web_search_service_cls
        self.prompt_service_cls = prompt_service_cls
        self.web_alignment_service_cls = web_alignment_service_cls
        self.capture_debug_trace = capture_debug_trace
        self.music_workflow_service = MusicWorkflowService(config)
        self._last_web_alignment: WebAlignmentResult | None = None
        self._last_web_attempts: list[WebSearchAttemptInfo] = []

    def ask(self, request: QueryRequest) -> QueryResponse:
        """Run the full question-answer flow and return structured UI-friendly results."""
        vector_store = self.vector_store_cls(self.config)
        ensure_index_compatible(vector_store)

        embedding_client = self.embedding_client_cls(self.config)
        retriever = self.retriever_cls(self.config, embedding_client, vector_store)
        chat_client = self.chat_client_cls(self.config)
        web_search_service = self.web_search_service_cls(self.config)
        prompt_service = self.prompt_service_cls()
        web_alignment_service = self.web_alignment_service_cls()
        workflow_plan = self.music_workflow_service.build_query_plan(request)
        self._last_web_alignment = None
        self._last_web_attempts = []

        logger.info("Retrieving relevant notes")
        if not self.capture_debug_trace:
            try:
                final_chunks = self._retrieve_chunks(retriever, request)
            except RuntimeError as exc:
                if request.retrieval_mode == RetrievalMode.LOCAL_ONLY:
                    raise
                final_chunks = []
                logger.info("Local retrieval unavailable; continuing with optional web fallback: %s", exc)
            primary_chunks = [chunk for chunk in final_chunks if not chunk.metadata.get("linked_context")]
            web_results, web_warnings = self._run_web_search_if_needed(
                request.question,
                primary_chunks=primary_chunks,
                retrieval_mode=request.retrieval_mode,
                web_search_service=web_search_service,
                web_alignment_service=web_alignment_service,
            )
            answer_result = _build_answer_result(
                workflow_plan.prompt_text,
                final_chunks,
                chat_client,
                prompt_service=prompt_service,
                web_results=web_results,
                retrieval_mode=request.retrieval_mode,
                answer_mode=request.answer_mode,
                local_retrieval_weak=_is_local_retrieval_weak(primary_chunks),
                domain_profile=request.domain_profile,
                collaboration_workflow=request.collaboration_workflow,
                workflow_input=request.workflow_input,
                web_alignment=self._last_web_alignment,
            )
            initial_candidates = []
            reranking_applied = bool(request.options.rerank or request.options.boost_tags)
            reranking_changed = False
        else:
            retrieval_debug = self._retrieve_chunks_with_debug(retriever, request)
            initial_candidates = retrieval_debug.initial_candidates
            primary_chunks = retrieval_debug.primary_chunks
            final_chunks = retrieval_debug.final_chunks
            web_results, web_warnings = self._run_web_search_if_needed(
                request.question,
                primary_chunks=primary_chunks,
                retrieval_mode=request.retrieval_mode,
                web_search_service=web_search_service,
                web_alignment_service=web_alignment_service,
            )
            answer_result = _build_answer_result(
                workflow_plan.prompt_text,
                final_chunks,
                chat_client,
                prompt_service=prompt_service,
                web_results=web_results,
                retrieval_mode=request.retrieval_mode,
                answer_mode=request.answer_mode,
                local_retrieval_weak=_is_local_retrieval_weak(primary_chunks),
                domain_profile=request.domain_profile,
                collaboration_workflow=request.collaboration_workflow,
                workflow_input=request.workflow_input,
                web_alignment=self._last_web_alignment,
            )
            reranking_applied = retrieval_debug.reranking_applied
            reranking_changed = retrieval_debug.reranking_changed

        warnings = _build_warnings(
            answer_result,
            web_results=web_results,
            answer_mode=request.answer_mode,
        )
        warnings.extend(web_warnings)
        alignment = getattr(self, "_last_web_alignment", None)
        attempts = list(self._last_web_attempts)
        warnings.extend(_build_guard_warnings(
            answer_result=answer_result,
            web_results=web_results,
            answer_mode=request.answer_mode,
            local_retrieval_weak=_is_local_retrieval_weak(primary_chunks),
            web_alignment=alignment,
            web_attempts=attempts,
        ))
        warnings = list(dict.fromkeys(warnings))
        linked_chunks = [chunk for chunk in answer_result.retrieved_chunks if chunk.metadata.get("linked_context")]
        saved_path = None
        citation_sources, citation_labels = build_citation_sources(answer_result.retrieved_chunks, web_results)
        inference_used = answer_uses_inference(answer_result.answer)
        evidence_types_used = tuple(
            source_type
            for source_type, enabled in (
                ("local_note", bool(answer_result.retrieved_chunks)),
                ("web", bool(web_results)),
            )
            if enabled
        )
        trust_counts = _count_trust_categories(answer_result.retrieved_chunks)

        if request.auto_save or self.config.auto_save_answer:
            saved_path = save_answer(
                workflow_plan.save_path,
                request.question,
                answer_result,
                title_override=request.save_title,
                source_type="saved_answer",
                status="draft",
                indexed=False,
                domain_profile=request.domain_profile.value,
                workflow_type=request.collaboration_workflow.value,
                workflow_input=request.workflow_input.as_dict(),
            )
            logger.info("Saved answer to %s", saved_path)

        return QueryResponse(
            answer_result=answer_result,
            warnings=warnings,
            linked_context_chunks=linked_chunks,
            web_results=web_results,
            saved_path=saved_path,
            debug=QueryDebugInfo(
                initial_candidates=initial_candidates,
                primary_chunks=primary_chunks,
                reranking_applied=reranking_applied,
                reranking_changed=reranking_changed,
                retrieval_filters=request.filters,
                retrieval_options=request.options,
                retrieval_scope_requested=request.retrieval_scope,
                retrieval_mode_requested=request.retrieval_mode,
                retrieval_mode_used=_resolve_retrieval_mode_used(request.retrieval_mode, web_results),
                answer_mode_requested=request.answer_mode,
                answer_mode_used=request.answer_mode,
                local_retrieval_weak=_is_local_retrieval_weak(primary_chunks),
                web_used=bool(web_results),
                evidence_types_used=evidence_types_used,
                inference_used=inference_used,
                citation_labels=tuple(citation_labels),
                web_query_used=alignment.query if alignment else "",
                web_query_strategy=alignment.strategy if alignment else WebQueryStrategy.RAW_QUESTION,
                web_results_filtered_count=alignment.filtered_count if alignment else 0,
                web_alignment_warning=alignment.warning if alignment else "",
                web_attempts=attempts,
                web_failure_reason=_summarize_web_failure_reason(attempts),
                web_provider_returned_results=any(attempt.provider_returned_results for attempt in attempts),
                web_results_discarded_by_filter=any(
                    attempt.results_discarded_by_filter for attempt in attempts
                ),
                web_retry_used=any(attempt.retry_used for attempt in attempts),
                curated_knowledge_chunks=trust_counts["curated_knowledge"],
                imported_knowledge_chunks=trust_counts["imported_knowledge"],
                non_curated_note_chunks=trust_counts["non_curated_note"],
                generated_or_imported_chunks=trust_counts["generated_or_imported"],
                hallucination_guard_warnings=tuple(
                    _build_guard_warnings(
                        answer_result=answer_result,
                        web_results=web_results,
                        answer_mode=request.answer_mode,
                        local_retrieval_weak=_is_local_retrieval_weak(primary_chunks),
                        web_alignment=alignment,
                        web_attempts=attempts,
                    )
                ),
            ),
            domain_profile=request.domain_profile,
            collaboration_workflow=request.collaboration_workflow,
            workflow_input=request.workflow_input,
        )

    def save(
        self,
        question: str,
        answer_result: AnswerResult,
        *,
        title_override: str | None = None,
        existing_response: QueryResponse | None = None,
    ) -> QueryResponse:
        """Persist an existing answer result and return updated response info."""
        saved_path = save_answer(
            self.music_workflow_service.default_save_path(
                existing_response.collaboration_workflow
                if existing_response is not None
                else CollaborationWorkflow.GENERAL_ASK
            ),
            question,
            answer_result,
            title_override=title_override,
            source_type="saved_answer",
            status="draft",
            indexed=False,
            domain_profile=(
                existing_response.domain_profile.value
                if existing_response is not None
                else None
            ),
            workflow_type=(
                existing_response.collaboration_workflow.value
                if existing_response is not None
                else None
            ),
            workflow_input=(
                existing_response.workflow_input.as_dict()
                if existing_response is not None
                else None
            ),
        )
        logger.info("Saved answer to %s", saved_path)
        if existing_response is not None:
            return existing_response.with_saved_path(saved_path)
        return QueryResponse(
            answer_result=answer_result,
            warnings=_build_warnings(answer_result, web_results=[], answer_mode=AnswerMode.BALANCED),
            linked_context_chunks=[
                chunk for chunk in answer_result.retrieved_chunks if chunk.metadata.get("linked_context")
            ],
            saved_path=saved_path,
            web_results=[],
            debug=QueryDebugInfo(),
            domain_profile=(
                existing_response.domain_profile
                if existing_response is not None
                else DomainProfile.ELECTRONIC_MUSIC
            ),
            collaboration_workflow=(
                existing_response.collaboration_workflow
                if existing_response is not None
                else CollaborationWorkflow.GENERAL_ASK
            ),
            workflow_input=(
                existing_response.workflow_input
                if existing_response is not None
                else WorkflowInput()
            ),
        )

    def _retrieve_chunks(self, retriever: Retriever, request: QueryRequest) -> list[RetrievedChunk]:
        try:
            return retriever.retrieve(
                request.question,
                filters=request.filters,
                options=request.options,
                retrieval_scope=request.retrieval_scope,
            )
        except TypeError as exc:
            if "retrieval_scope" not in str(exc):
                raise
            return retriever.retrieve(
                request.question,
                filters=request.filters,
                options=request.options,
            )

    def _retrieve_chunks_with_debug(self, retriever: Retriever, request: QueryRequest):
        try:
            return retriever.retrieve_with_debug(
                request.question,
                filters=request.filters,
                options=request.options,
                retrieval_scope=request.retrieval_scope,
            )
        except TypeError as exc:
            if "retrieval_scope" not in str(exc):
                raise
            return retriever.retrieve_with_debug(
                request.question,
                filters=request.filters,
                options=request.options,
            )

    def _run_web_search_if_needed(
        self,
        question: str,
        *,
        primary_chunks: list[RetrievedChunk],
        retrieval_mode: RetrievalMode,
        web_search_service: WebSearchService,
        web_alignment_service: WebAlignmentService,
    ) -> tuple[list[WebSearchResult], list[str]]:
        should_use_web = _should_use_web_search(retrieval_mode, primary_chunks)
        if not should_use_web:
            self._last_web_alignment = None
            self._last_web_attempts = []
            return [], []
        attempts: list[WebSearchAttemptInfo] = []

        first_query, first_strategy, _ = web_alignment_service.build_query(
            question,
            primary_chunks=primary_chunks,
            retrieval_mode=retrieval_mode,
            provider=self.config.web_search_provider,
        )
        first_results, first_warnings, first_alignment, first_attempt = self._perform_web_attempt(
            question=question,
            query=first_query,
            strategy=first_strategy,
            primary_chunks=primary_chunks,
            retrieval_mode=retrieval_mode,
            web_search_service=web_search_service,
            web_alignment_service=web_alignment_service,
            retry_used=False,
        )
        attempts.append(first_attempt)

        if first_results:
            self._last_web_alignment = first_alignment
            self._last_web_attempts = attempts
            return first_results, first_warnings

        retry_query, retry_strategy, _ = web_alignment_service.build_retry_query(
            question,
            primary_chunks=primary_chunks,
            provider=self.config.web_search_provider,
        )
        retry_needed = (
            retry_query != first_query
            and first_attempt.failure_reason in {
                "provider_returned_no_results",
                "provider_returned_no_usable_results",
            }
        )
        if not retry_needed:
            self._last_web_alignment = first_alignment
            self._last_web_attempts = attempts
            return [], first_warnings

        retry_results, retry_warnings, retry_alignment, retry_attempt = self._perform_web_attempt(
            question=question,
            query=retry_query,
            strategy=retry_strategy,
            primary_chunks=primary_chunks,
            retrieval_mode=retrieval_mode,
            web_search_service=web_search_service,
            web_alignment_service=web_alignment_service,
            retry_used=True,
        )
        attempts.append(retry_attempt)
        self._last_web_alignment = retry_alignment or first_alignment
        self._last_web_attempts = attempts

        if retry_results:
            return retry_results, retry_warnings

        combined_warnings = first_warnings + retry_warnings
        if retry_attempt.failure_reason:
            combined_warnings.append(
                "A lighter retry query was attempted but still produced no usable aligned web evidence."
            )
        return [], list(dict.fromkeys(combined_warnings))

    def _perform_web_attempt(
        self,
        *,
        question: str,
        query: str,
        strategy: WebQueryStrategy,
        primary_chunks: list[RetrievedChunk],
        retrieval_mode: RetrievalMode,
        web_search_service: WebSearchService,
        web_alignment_service: WebAlignmentService,
        retry_used: bool,
    ) -> tuple[list[WebSearchResult], list[str], WebAlignmentResult | None, WebSearchAttemptInfo]:
        warnings: list[str] = []
        try:
            provider_results = web_search_service.search(query)
        except Exception as exc:
            return [], [f"Web search was requested but unavailable: {exc}"], None, WebSearchAttemptInfo(
                query=query,
                strategy=strategy,
                retry_used=retry_used,
                failure_reason="provider_error",
                outcome="provider_error",
            )

        alignment = web_alignment_service.build_alignment(
            question,
            primary_chunks=primary_chunks,
            web_results=provider_results,
            retrieval_mode=retrieval_mode,
            provider=self.config.web_search_provider,
        )
        filtered_results = alignment.filtered_results
        filtered_count = alignment.filtered_count
        provider_returned_results = bool(provider_results)
        if alignment.warning:
            warnings.append(alignment.warning)

        if not provider_results:
            warnings.append(_no_web_results_warning(strategy, retry_used))
            return [], warnings, alignment, WebSearchAttemptInfo(
                query=query,
                strategy=strategy,
                retry_used=retry_used,
                provider_returned_results=False,
                provider_result_count=0,
                usable_result_count=0,
                filtered_count=0,
                failure_reason="provider_returned_no_results",
                outcome="no_provider_results",
            )

        if not filtered_results:
            warnings.append(_filtered_out_warning(strategy, retry_used))
            return [], warnings, alignment, WebSearchAttemptInfo(
                query=query,
                strategy=strategy,
                retry_used=retry_used,
                provider_returned_results=provider_returned_results,
                provider_result_count=len(provider_results),
                usable_result_count=0,
                filtered_count=filtered_count,
                results_discarded_by_filter=True,
                failure_reason="all_results_filtered_out",
                outcome="filtered_out",
            )

        return filtered_results, warnings, alignment, WebSearchAttemptInfo(
            query=query,
            strategy=strategy,
            retry_used=retry_used,
            provider_returned_results=provider_returned_results,
            provider_result_count=len(provider_results),
            usable_result_count=len(filtered_results),
            filtered_count=filtered_count,
            results_discarded_by_filter=filtered_count > 0,
            outcome="usable_results",
        )


def _build_answer_result(
    question: str,
    chunks: list[RetrievedChunk],
    chat_client: OllamaChatClient,
    *,
    prompt_service: PromptService,
    web_results: list[WebSearchResult] | None = None,
    retrieval_mode: RetrievalMode = RetrievalMode.LOCAL_ONLY,
    answer_mode: AnswerMode = AnswerMode.BALANCED,
    local_retrieval_weak: bool = False,
    domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC,
    collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
    workflow_input: WorkflowInput | None = None,
    web_alignment: WebAlignmentResult | None = None,
) -> AnswerResult:
    web_results = web_results or []
    if not chunks and not web_results:
        return AnswerResult(
            answer=_insufficient_evidence_message(answer_mode),
            sources=[],
            retrieved_chunks=[],
        )

    if answer_mode == AnswerMode.STRICT and local_retrieval_weak and not web_results:
        citation_sources, _ = build_citation_sources(chunks, web_results)
        answer = _insufficient_evidence_message(answer_mode)
        if citation_sources:
            answer = f"{answer}\n\nAvailable evidence:\n" + "\n".join(citation_sources)
        return AnswerResult(answer=answer, sources=citation_sources, retrieved_chunks=chunks)

    prompt_payload = prompt_service.build_prompt_payload(
        question,
        chunks,
        web_results=web_results or [],
        retrieval_mode=retrieval_mode,
        answer_mode=answer_mode,
        local_retrieval_weak=local_retrieval_weak,
        domain_profile=domain_profile,
        collaboration_workflow=collaboration_workflow,
        workflow_input=workflow_input,
        web_query_used=web_alignment.query if web_alignment else question,
        web_query_strategy=web_alignment.strategy.value if web_alignment else "raw_question",
        web_alignment_note=web_alignment.warning if web_alignment else "",
    )
    answer = chat_client.answer_with_prompt(prompt_payload)
    answer = enforce_citation_summary(answer, prompt_payload.citation_labels, answer_mode)
    sources, _ = build_citation_sources(chunks, web_results)

    return AnswerResult(answer=answer, sources=sources, retrieved_chunks=chunks)


def _build_warnings(
    answer_result: AnswerResult,
    *,
    web_results: list[WebSearchResult],
    answer_mode: AnswerMode,
) -> list[str]:
    warnings: list[str] = []
    if not answer_result.retrieved_chunks and not web_results:
        warnings.append(_insufficient_evidence_message(answer_mode))
        return warnings

    if not answer_result.retrieved_chunks and web_results:
        warnings.append("No relevant local note context was retrieved; this answer uses external web evidence.")
        return warnings

    primary_chunks = [chunk for chunk in answer_result.retrieved_chunks if not chunk.metadata.get("linked_context")]
    if not primary_chunks:
        if web_results:
            warnings.append(
                "No directly retrieved local chunks were used; this answer relies on linked-note and web evidence."
            )
        else:
            warnings.append("No directly retrieved chunks were used; only linked context is available.")

    distances = [
        chunk.distance_or_score
        for chunk in primary_chunks
        if chunk.distance_or_score is not None
    ]
    if distances and min(distances) > 0.7:
        if web_results:
            warnings.append("Local retrieval may be weak; external web evidence was used to supplement the answer.")
        else:
            warnings.append("Retrieval may be weak; the closest chunk distance is relatively high.")

    return warnings


def _build_guard_warnings(
    *,
    answer_result: AnswerResult,
    web_results: list[WebSearchResult],
    answer_mode: AnswerMode,
    local_retrieval_weak: bool,
    web_alignment: WebAlignmentResult | None,
    web_attempts: list[WebSearchAttemptInfo],
) -> list[str]:
    warnings: list[str] = []
    if answer_mode == AnswerMode.STRICT and local_retrieval_weak and not web_results:
        warnings.append("Strict mode limited the answer because the available evidence was weak.")
    if web_results and not answer_result.retrieved_chunks:
        warnings.append("This answer is based on external web evidence rather than local notes.")
    if web_results and answer_result.retrieved_chunks:
        warnings.append("This answer combines local notes with external web evidence; sources are labeled separately.")
    if web_alignment and web_alignment.strategy.value == "local_guided" and web_alignment.query:
        warnings.append("Hybrid web search was guided by the strongest local note topics.")
    if local_retrieval_weak and web_attempts and not web_results:
        warnings.append("Local retrieval was weak, and web search still did not produce usable aligned evidence.")
    return warnings


def _no_web_results_warning(strategy: WebQueryStrategy, retry_used: bool) -> str:
    if retry_used:
        return "The lighter retry web query returned no provider results."
    if strategy == WebQueryStrategy.LOCAL_GUIDED:
        return "The local-guided web query returned no provider results."
    return "The web query returned no provider results."


def _filtered_out_warning(strategy: WebQueryStrategy, retry_used: bool) -> str:
    if retry_used:
        return "The lighter retry web query returned results, but they were filtered out as off-topic."
    if strategy == WebQueryStrategy.LOCAL_GUIDED:
        return "The local-guided web query returned results, but they were filtered out as off-topic."
    return "The web query returned results, but they were filtered out as off-topic."


def _summarize_web_failure_reason(web_attempts: list[WebSearchAttemptInfo]) -> str:
    if not web_attempts:
        return ""
    last_attempt = web_attempts[-1]
    if last_attempt.failure_reason == "provider_error":
        return "provider_error"
    if last_attempt.failure_reason == "all_results_filtered_out":
        return "all_results_filtered_out"
    if last_attempt.failure_reason == "provider_returned_no_results":
        return "provider_returned_no_results"
    return ""


def _insufficient_evidence_message(answer_mode: AnswerMode) -> str:
    if answer_mode == AnswerMode.STRICT:
        return "Insufficient evidence to answer confidently from the retrieved sources."
    return "No relevant local note context or external web evidence matched the current retrieval mode."


def _should_use_web_search(retrieval_mode: RetrievalMode, primary_chunks: list[RetrievedChunk]) -> bool:
    if retrieval_mode == RetrievalMode.LOCAL_ONLY:
        return False
    if retrieval_mode == RetrievalMode.HYBRID:
        return True
    return not primary_chunks or _is_local_retrieval_weak(primary_chunks)


def _is_local_retrieval_weak(primary_chunks: list[RetrievedChunk]) -> bool:
    if not primary_chunks:
        return True
    distances = [chunk.distance_or_score for chunk in primary_chunks if chunk.distance_or_score is not None]
    return bool(distances) and min(distances) > 0.7


def _resolve_retrieval_mode_used(
    retrieval_mode: RetrievalMode,
    web_results: list[WebSearchResult],
) -> RetrievalModeUsed:
    if retrieval_mode == RetrievalMode.LOCAL_ONLY:
        return RetrievalModeUsed.LOCAL_ONLY
    if retrieval_mode == RetrievalMode.HYBRID:
        return RetrievalModeUsed.HYBRID if web_results else RetrievalModeUsed.HYBRID_NO_WEB_RESULTS
    return RetrievalModeUsed.AUTO_WITH_WEB if web_results else RetrievalModeUsed.AUTO_LOCAL_ONLY


def _count_trust_categories(chunks: list[RetrievedChunk]) -> dict[str, int]:
    counts = {
        "curated_knowledge": 0,
        "imported_knowledge": 0,
        "non_curated_note": 0,
        "generated_or_imported": 0,
    }
    for chunk in chunks:
        category = str(chunk.metadata.get("content_category", "")).strip().lower()
        if category in counts:
            counts[category] += 1
    return counts
