"""Retrieve relevant note chunks for a user query."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import AppConfig
from model_clients import EmbeddingClient
from reranker import rerank_chunks
from services.models import (
    CollaborationWorkflow,
    DomainProfile,
    RetrievalScoreDebug,
    RetrievalScope,
    TrackContext,
)
from utils import RetrievalFilters, RetrievalOptions, RetrievedChunk
from vector_store import VectorStore


@dataclass(slots=True)
class RetrievalDebugResult:
    """Public retrieval result with debug-friendly intermediate stages."""

    initial_candidates: list[RetrievedChunk]
    reranked_candidates: list[RetrievedChunk]
    primary_chunks: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]
    reranking_applied: bool
    reranking_changed: bool
    reranking_details: list[RetrievalScoreDebug] = field(default_factory=list)


class Retriever:
    """Coordinates query embedding and vector search."""

    def __init__(
        self,
        config: AppConfig,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        options: RetrievalOptions | None = None,
        retrieval_scope: RetrievalScope = RetrievalScope.KNOWLEDGE,
        track_context: TrackContext | None = None,
        collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.GENERAL_ASK,
        section_focus: str | None = None,
        domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC,
    ) -> list[RetrievedChunk]:
        """Return the top-k relevant chunks for a question."""
        return self.retrieve_with_debug(
            query,
            filters=filters,
            options=options,
            retrieval_scope=retrieval_scope,
            track_context=track_context,
            collaboration_workflow=collaboration_workflow,
            section_focus=section_focus,
            domain_profile=domain_profile,
        ).final_chunks

    def retrieve_with_debug(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        options: RetrievalOptions | None = None,
        retrieval_scope: RetrievalScope = RetrievalScope.KNOWLEDGE,
        track_context: TrackContext | None = None,
        collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.GENERAL_ASK,
        section_focus: str | None = None,
        domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC,
    ) -> RetrievalDebugResult:
        """Return retrieved chunks plus public intermediate retrieval details."""
        if self.vector_store.count() == 0:
            raise RuntimeError("The vector store is empty. Run `python main.py index` first.")

        settings = self._resolve_settings(options)
        candidates = self._run_vector_retrieval(
            query,
            filters,
            settings["candidate_count"],
            retrieval_scope,
            settings["include_saved_answers"],
        )
        ranked_chunks, reranking_details = self._apply_reranking(
            query,
            candidates,
            settings,
            track_context=track_context,
            collaboration_workflow=collaboration_workflow,
            section_focus=section_focus,
            domain_profile=domain_profile,
        )
        primary_chunks = self._select_primary_chunks(ranked_chunks, settings["top_k"])
        final_chunks = self._expand_linked_chunks(primary_chunks, settings["include_linked_notes"], retrieval_scope)
        reranking_applied = bool(
            settings["rerank_enabled"]
            or settings["boost_tags"]
            or track_context is not None
            or bool((section_focus or "").strip())
            or collaboration_workflow != CollaborationWorkflow.GENERAL_ASK
        )
        return RetrievalDebugResult(
            initial_candidates=candidates,
            reranked_candidates=ranked_chunks,
            primary_chunks=primary_chunks,
            final_chunks=final_chunks,
            reranking_applied=reranking_applied,
            reranking_changed=_chunk_signatures(candidates) != _chunk_signatures(ranked_chunks),
            reranking_details=reranking_details,
        )

    def _resolve_settings(self, options: RetrievalOptions | None) -> dict[str, object]:
        top_k = options.top_k if options and options.top_k is not None else self.config.top_k_results
        candidate_count = (
            options.candidate_count
            if options and options.candidate_count is not None
            else max(top_k, top_k * self.config.retrieval_candidate_multiplier)
        )
        rerank_enabled = (
            options.rerank
            if options and options.rerank is not None
            else self.config.enable_reranking
        )
        include_linked_notes = (
            options.include_linked_notes
            if options and options.include_linked_notes is not None
            else self.config.enable_linked_note_expansion
        )
        include_saved_answers = (
            options.include_saved_answers
            if options and options.include_saved_answers is not None
            else True
        )
        boost_tags = options.boost_tags if options else ()
        return {
            "top_k": top_k,
            "candidate_count": candidate_count,
            "rerank_enabled": rerank_enabled,
            "include_linked_notes": include_linked_notes,
            "include_saved_answers": include_saved_answers,
            "boost_tags": boost_tags,
        }

    def _run_vector_retrieval(
        self,
        query: str,
        filters: RetrievalFilters | None,
        candidate_count: int,
        retrieval_scope: RetrievalScope,
        include_saved_answers: bool,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_client.embed_text(query)
        try:
            return self.vector_store.query(
                query_embedding,
                candidate_count,
                filters=filters,
                retrieval_scope=retrieval_scope.value,
                include_saved_answers=include_saved_answers,
            )
        except TypeError as exc:
            message = str(exc)
            if "include_saved_answers" not in message and "retrieval_scope" not in message:
                raise
            try:
                return self.vector_store.query(
                    query_embedding,
                    candidate_count,
                    filters=filters,
                    include_saved_answers=include_saved_answers,
                )
            except TypeError as inner_exc:
                inner_message = str(inner_exc)
                if "retrieval_scope" not in inner_message and "include_saved_answers" not in inner_message:
                    raise
                return self.vector_store.query(
                    query_embedding,
                    candidate_count,
                    filters=filters,
                )

    def _apply_reranking(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        settings: dict[str, object],
        *,
        track_context: TrackContext | None,
        collaboration_workflow: CollaborationWorkflow,
        section_focus: str | None,
        domain_profile: DomainProfile,
    ) -> tuple[list[RetrievedChunk], list[RetrievalScoreDebug]]:
        if (
            not settings["rerank_enabled"]
            and not settings["boost_tags"]
            and track_context is None
            and not (section_focus or "").strip()
            and collaboration_workflow == CollaborationWorkflow.GENERAL_ASK
        ):
            return chunks, []
        ranked_chunks, details = rerank_chunks(
            query,
            chunks,
            boost_tags=settings["boost_tags"],
            tag_boost_weight=self.config.tag_boost_weight,
            track_context=track_context,
            collaboration_workflow=collaboration_workflow,
            section_focus=section_focus,
            domain_profile=domain_profile,
        )
        return ranked_chunks, details

    def _select_primary_chunks(self, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]

    def _expand_linked_chunks(
        self,
        primary_chunks: list[RetrievedChunk],
        include_linked_notes: bool,
        retrieval_scope: RetrievalScope,
    ) -> list[RetrievedChunk]:
        if not include_linked_notes:
            return primary_chunks

        linked_note_keys = _collect_linked_note_keys(primary_chunks)
        if not linked_note_keys:
            return primary_chunks

        primary_note_keys = {
            str(chunk.metadata.get("note_key"))
            for chunk in primary_chunks
            if chunk.metadata.get("note_key")
        }
        try:
            linked_chunks = self.vector_store.get_chunks_by_note_keys(
                linked_note_keys[: self.config.max_linked_notes],
                max_chunks_per_note=self.config.linked_note_chunks_per_note,
                retrieval_scope=retrieval_scope.value,
                excluded_note_keys=primary_note_keys,
            )
        except TypeError as exc:
            if "retrieval_scope" not in str(exc):
                raise
            linked_chunks = self.vector_store.get_chunks_by_note_keys(
                linked_note_keys[: self.config.max_linked_notes],
                max_chunks_per_note=self.config.linked_note_chunks_per_note,
                excluded_note_keys=primary_note_keys,
            )
        return primary_chunks + linked_chunks


def _collect_linked_note_keys(chunks: list[RetrievedChunk]) -> list[str]:
    linked_keys: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        serialized = chunk.metadata.get("linked_note_keys_serialized", "")
        if not isinstance(serialized, str) or not serialized:
            continue
        for note_key in serialized.split("|"):
            if not note_key or note_key in seen:
                continue
            seen.add(note_key)
            linked_keys.append(note_key)

    return linked_keys


def _chunk_signatures(chunks: list[RetrievedChunk]) -> list[tuple[object, object, object]]:
    return [
        (
            chunk.metadata.get("source_path"),
            chunk.metadata.get("chunk_index"),
            chunk.metadata.get("note_title"),
        )
        for chunk in chunks
    ]
