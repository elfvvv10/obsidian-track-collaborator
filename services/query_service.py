"""Thin query service built on existing retrieval and answer modules."""

from __future__ import annotations

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from retriever import Retriever
from saver import save_answer
from services.common import ensure_index_compatible
from services.models import QueryDebugInfo, QueryRequest, QueryResponse
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
        capture_debug_trace: bool = True,
    ) -> None:
        self.config = config
        self.embedding_client_cls = embedding_client_cls
        self.chat_client_cls = chat_client_cls
        self.retriever_cls = retriever_cls
        self.vector_store_cls = vector_store_cls
        self.web_search_service_cls = web_search_service_cls
        self.capture_debug_trace = capture_debug_trace

    def ask(self, request: QueryRequest) -> QueryResponse:
        """Run the full question-answer flow and return structured UI-friendly results."""
        vector_store = self.vector_store_cls(self.config)
        ensure_index_compatible(vector_store)

        embedding_client = self.embedding_client_cls(self.config)
        retriever = self.retriever_cls(self.config, embedding_client, vector_store)
        chat_client = self.chat_client_cls(self.config)
        web_search_service = self.web_search_service_cls(self.config)

        logger.info("Retrieving relevant notes")
        if not self.capture_debug_trace:
            try:
                final_chunks = retriever.retrieve(
                    request.question,
                    filters=request.filters,
                    options=request.options,
                )
            except RuntimeError as exc:
                if request.retrieval_mode == "local_only":
                    raise
                final_chunks = []
                logger.info("Local retrieval unavailable; continuing with optional web fallback: %s", exc)
            web_results, web_warnings = self._run_web_search_if_needed(
                request.question,
                primary_chunks=[chunk for chunk in final_chunks if not chunk.metadata.get("linked_context")],
                retrieval_mode=request.retrieval_mode,
                web_search_service=web_search_service,
            )
            answer_result = _build_answer_result(
                request.question,
                final_chunks,
                chat_client,
                web_results=web_results,
                retrieval_mode=request.retrieval_mode,
            )
            initial_candidates = []
            primary_chunks = [chunk for chunk in final_chunks if not chunk.metadata.get("linked_context")]
            reranking_applied = bool(request.options.rerank or request.options.boost_tags)
            reranking_changed = False
        else:
            retrieval_settings = retriever._resolve_settings(request.options)
            initial_candidates = retriever._run_vector_retrieval(
                request.question,
                request.filters,
                int(retrieval_settings["candidate_count"]),
            )
            reranked_candidates = retriever._apply_reranking(
                request.question,
                initial_candidates,
                retrieval_settings,
            )
            primary_chunks = retriever._select_primary_chunks(
                reranked_candidates,
                int(retrieval_settings["top_k"]),
            )
            final_chunks = retriever._expand_linked_chunks(
                primary_chunks,
                bool(retrieval_settings["include_linked_notes"]),
            )
            web_results, web_warnings = self._run_web_search_if_needed(
                request.question,
                primary_chunks=primary_chunks,
                retrieval_mode=request.retrieval_mode,
                web_search_service=web_search_service,
            )
            answer_result = _build_answer_result(
                request.question,
                final_chunks,
                chat_client,
                web_results=web_results,
                retrieval_mode=request.retrieval_mode,
            )
            reranking_applied = bool(retrieval_settings["rerank_enabled"] or retrieval_settings["boost_tags"])
            reranking_changed = _chunk_signatures(initial_candidates) != _chunk_signatures(reranked_candidates)

        warnings = _build_warnings(answer_result)
        warnings.extend(web_warnings)
        linked_chunks = [chunk for chunk in answer_result.retrieved_chunks if chunk.metadata.get("linked_context")]
        saved_path = None

        if request.auto_save or self.config.auto_save_answer:
            saved_path = save_answer(
                self.config.obsidian_output_path,
                request.question,
                answer_result,
                title_override=request.save_title,
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
                retrieval_mode_requested=request.retrieval_mode,
                retrieval_mode_used=_resolve_retrieval_mode_used(request.retrieval_mode, web_results),
                local_retrieval_weak=_is_local_retrieval_weak(primary_chunks),
                web_used=bool(web_results),
            ),
        )

    def save(
        self,
        question: str,
        answer_result: AnswerResult,
        *,
        title_override: str | None = None,
    ) -> QueryResponse:
        """Persist an existing answer result and return updated response info."""
        saved_path = save_answer(
            self.config.obsidian_output_path,
            question,
            answer_result,
            title_override=title_override,
        )
        logger.info("Saved answer to %s", saved_path)
        return QueryResponse(
            answer_result=answer_result,
            warnings=_build_warnings(answer_result),
            linked_context_chunks=[
                chunk for chunk in answer_result.retrieved_chunks if chunk.metadata.get("linked_context")
            ],
            saved_path=saved_path,
            web_results=[],
            debug=QueryDebugInfo(),
        )

    def _run_web_search_if_needed(
        self,
        question: str,
        *,
        primary_chunks: list[RetrievedChunk],
        retrieval_mode: str,
        web_search_service: WebSearchService,
    ) -> tuple[list[WebSearchResult], list[str]]:
        should_use_web = _should_use_web_search(retrieval_mode, primary_chunks)
        if not should_use_web:
            return [], []

        try:
            results = web_search_service.search(question)
        except Exception as exc:
            return [], [f"Web search was requested but unavailable: {exc}"]

        if not results:
            return [], ["Web search did not return any useful external results."]
        return results, []


def _build_answer_result(
    question: str,
    chunks: list[RetrievedChunk],
    chat_client: OllamaChatClient,
    *,
    web_results: list[WebSearchResult] | None = None,
    retrieval_mode: str = "local_only",
) -> AnswerResult:
    web_results = web_results or []
    if not chunks and not web_results:
        return AnswerResult(
            answer="No relevant local note context or external web evidence matched the current retrieval mode.",
            sources=[],
            retrieved_chunks=[],
        )

    answer = chat_client.answer_question(
        question,
        chunks,
        web_results=web_results,
        retrieval_mode=retrieval_mode,
    )
    seen_sources: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        source = (
            f"[Local] {chunk.metadata.get('note_title', 'Untitled')} "
            f"({chunk.metadata.get('source_path', 'unknown')})"
        )
        if source in seen_sources:
            continue
        seen_sources.add(source)
        sources.append(source)
    for result in web_results:
        source = f"[Web] {result.title} ({result.url})"
        if source in seen_sources:
            continue
        seen_sources.add(source)
        sources.append(source)

    return AnswerResult(answer=answer, sources=sources, retrieved_chunks=chunks)


def _build_warnings(answer_result: AnswerResult) -> list[str]:
    warnings: list[str] = []
    if not answer_result.retrieved_chunks:
        warnings.append("No relevant note context was retrieved.")
        return warnings

    primary_chunks = [chunk for chunk in answer_result.retrieved_chunks if not chunk.metadata.get("linked_context")]
    if not primary_chunks:
        warnings.append("No directly retrieved chunks were used; only linked context is available.")

    distances = [
        chunk.distance_or_score
        for chunk in primary_chunks
        if chunk.distance_or_score is not None
    ]
    if distances and min(distances) > 0.7:
        warnings.append("Retrieval may be weak; the closest chunk distance is relatively high.")

    return warnings


def _should_use_web_search(retrieval_mode: str, primary_chunks: list[RetrievedChunk]) -> bool:
    if retrieval_mode == "local_only":
        return False
    if retrieval_mode == "hybrid":
        return True
    return not primary_chunks or _is_local_retrieval_weak(primary_chunks)


def _is_local_retrieval_weak(primary_chunks: list[RetrievedChunk]) -> bool:
    if not primary_chunks:
        return True
    distances = [chunk.distance_or_score for chunk in primary_chunks if chunk.distance_or_score is not None]
    return bool(distances) and min(distances) > 0.7


def _resolve_retrieval_mode_used(retrieval_mode: str, web_results: list[WebSearchResult]) -> str:
    if retrieval_mode == "local_only":
        return "local_only"
    if retrieval_mode == "hybrid":
        return "hybrid" if web_results else "hybrid_no_web_results"
    return "auto_with_web" if web_results else "auto_local_only"


def _chunk_signatures(chunks: list[RetrievedChunk]) -> list[tuple[object, object, object]]:
    return [
        (
            chunk.metadata.get("source_path"),
            chunk.metadata.get("chunk_index"),
            chunk.metadata.get("note_title"),
        )
        for chunk in chunks
    ]
