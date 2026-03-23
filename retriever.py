"""Retrieve relevant note chunks for a user query."""

from __future__ import annotations

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from reranker import rerank_chunks
from utils import RetrievalFilters, RetrievalOptions, RetrievedChunk
from vector_store import VectorStore


class Retriever:
    """Coordinates query embedding and vector search."""

    def __init__(
        self,
        config: AppConfig,
        embedding_client: OllamaEmbeddingClient,
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
    ) -> list[RetrievedChunk]:
        """Return the top-k relevant chunks for a question."""
        if self.vector_store.count() == 0:
            raise RuntimeError("The vector store is empty. Run `python main.py index` first.")

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
        boost_tags = options.boost_tags if options else ()
        include_linked_notes = (
            options.include_linked_notes
            if options and options.include_linked_notes is not None
            else self.config.enable_linked_note_expansion
        )

        query_embedding = self.embedding_client.embed_text(query)
        chunks = self.vector_store.query(query_embedding, candidate_count, filters=filters)
        if rerank_enabled or boost_tags:
            chunks = rerank_chunks(
                query,
                chunks,
                boost_tags=boost_tags,
                tag_boost_weight=self.config.tag_boost_weight,
            )
        primary_chunks = chunks[:top_k]
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
