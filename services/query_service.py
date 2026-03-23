"""Thin query service built on existing retrieval and answer modules."""

from __future__ import annotations

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from retriever import Retriever
from saver import save_answer
from services.common import ensure_index_compatible
from services.models import QueryDebugInfo, QueryRequest, QueryResponse
from utils import AnswerResult, RetrievedChunk, get_logger
from vector_store import VectorStore


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
        capture_debug_trace: bool = True,
    ) -> None:
        self.config = config
        self.embedding_client_cls = embedding_client_cls
        self.chat_client_cls = chat_client_cls
        self.retriever_cls = retriever_cls
        self.vector_store_cls = vector_store_cls
        self.capture_debug_trace = capture_debug_trace

    def ask(self, request: QueryRequest) -> QueryResponse:
        """Run the full question-answer flow and return structured UI-friendly results."""
        vector_store = self.vector_store_cls(self.config)
        ensure_index_compatible(vector_store)

        embedding_client = self.embedding_client_cls(self.config)
        retriever = self.retriever_cls(self.config, embedding_client, vector_store)
        chat_client = self.chat_client_cls(self.config)

        logger.info("Retrieving relevant notes")
        if not self.capture_debug_trace:
            final_chunks = retriever.retrieve(
                request.question,
                filters=request.filters,
                options=request.options,
            )
            answer_result = _build_answer_result(
                request.question,
                final_chunks,
                chat_client,
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
            answer_result = _build_answer_result(
                request.question,
                final_chunks,
                chat_client,
            )
            reranking_applied = bool(retrieval_settings["rerank_enabled"] or retrieval_settings["boost_tags"])
            reranking_changed = _chunk_signatures(initial_candidates) != _chunk_signatures(reranked_candidates)

        warnings = _build_warnings(answer_result)
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
            saved_path=saved_path,
            debug=QueryDebugInfo(
                initial_candidates=initial_candidates,
                primary_chunks=primary_chunks,
                reranking_applied=reranking_applied,
                reranking_changed=reranking_changed,
                retrieval_filters=request.filters,
                retrieval_options=request.options,
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
            debug=QueryDebugInfo(),
        )


def _build_answer_result(
    question: str,
    chunks: list[RetrievedChunk],
    chat_client: OllamaChatClient,
) -> AnswerResult:
    if not chunks:
        return AnswerResult(
            answer="No relevant note context matched the current retrieval filters.",
            sources=[],
            retrieved_chunks=[],
        )

    answer = chat_client.answer_question(question, chunks)
    seen_sources: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        source = f"{chunk.metadata.get('note_title', 'Untitled')} ({chunk.metadata.get('source_path', 'unknown')})"
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


def _chunk_signatures(chunks: list[RetrievedChunk]) -> list[tuple[object, object, object]]:
    return [
        (
            chunk.metadata.get("source_path"),
            chunk.metadata.get("chunk_index"),
            chunk.metadata.get("note_title"),
        )
        for chunk in chunks
    ]
