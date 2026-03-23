"""ChromaDB persistence helpers."""

from __future__ import annotations

import chromadb
from math import sqrt

from config import AppConfig
from utils import Chunk, RetrievalFilters, RetrievedChunk


class VectorStore:
    """Wrapper around a persistent ChromaDB collection."""

    def __init__(self, config: AppConfig) -> None:
        self.client = chromadb.PersistentClient(path=str(config.chroma_db_path))
        self.collection_name = config.chroma_collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def reset(self) -> None:
        """Delete and recreate the configured collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunk texts, embeddings, and metadata."""
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_path": chunk.source_path,
                    "source_path_normalized": chunk.source_path,
                    "source_dir": chunk.source_dir,
                    "note_title": chunk.note_title,
                    "chunk_index": chunk.chunk_index,
                    "heading_context": chunk.heading_context,
                    "note_key": chunk.note_key,
                    "note_fingerprint": chunk.note_fingerprint,
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Query the vector store and return retrieved chunks."""
        if filters and filters.path_contains:
            return self._query_with_path_contains(query_embedding, top_k, filters)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=self._build_where(filters),
        )

        return self._to_retrieved_chunks(results)

    def list_note_fingerprints(self) -> dict[str, str]:
        """Return the latest known note fingerprints keyed by note key."""
        results = self.collection.get(include=["metadatas"])
        fingerprints: dict[str, str] = {}

        for metadata in results.get("metadatas", []):
            if not metadata:
                continue
            note_key = metadata.get("note_key")
            note_fingerprint = metadata.get("note_fingerprint")
            if isinstance(note_key, str) and isinstance(note_fingerprint, str):
                fingerprints[note_key] = note_fingerprint
        return fingerprints

    def delete_by_note_keys(self, note_keys: list[str]) -> None:
        """Delete all chunks associated with the provided note keys."""
        for note_key in note_keys:
            self.collection.delete(where={"note_key": note_key})

    def get_all_chunks(
        self,
        filters: RetrievalFilters | None = None,
    ) -> list[tuple[str, dict[str, object], list[float]]]:
        """Return all chunk documents, metadata, and embeddings for filtered search."""
        results = self.collection.get(
            include=["documents", "metadatas", "embeddings"],
            where=self._build_where(filters),
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings", [])
        chunk_rows: list[tuple[str, dict[str, object], list[float]]] = []
        for document, metadata, embedding in zip(documents, metadatas, embeddings):
            if not document or not metadata or embedding is None:
                continue
            chunk_rows.append((document, dict(metadata), list(embedding)))
        return chunk_rows

    def _query_with_path_contains(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: RetrievalFilters,
    ) -> list[RetrievedChunk]:
        normalized_substring = filters.path_contains.lower()
        candidates = self.get_all_chunks(filters=RetrievalFilters(folder=filters.folder))

        filtered_candidates: list[RetrievedChunk] = []
        for document, metadata, embedding in candidates:
            source_path = str(metadata.get("source_path_normalized", "")).lower()
            if normalized_substring not in source_path:
                continue
            filtered_candidates.append(
                RetrievedChunk(
                    text=document,
                    metadata=metadata,
                    distance_or_score=_cosine_distance(query_embedding, embedding),
                )
            )

        filtered_candidates.sort(
            key=lambda chunk: float("inf")
            if chunk.distance_or_score is None
            else chunk.distance_or_score
        )
        return filtered_candidates[:top_k]

    def _build_where(self, filters: RetrievalFilters | None) -> dict[str, object] | None:
        if not filters or not filters.folder:
            return None
        return {"source_dir": filters.folder}

    def _to_retrieved_chunks(self, results: dict[str, object]) -> list[RetrievedChunk]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not document or not metadata:
                continue
            retrieved.append(
                RetrievedChunk(
                    text=document,
                    metadata=dict(metadata),
                    distance_or_score=distance,
                )
            )
        return retrieved

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()


def _cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = numerator / (left_norm * right_norm)
    return 1 - similarity
