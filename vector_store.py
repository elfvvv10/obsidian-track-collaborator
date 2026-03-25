"""ChromaDB persistence helpers."""

from __future__ import annotations

import chromadb
from math import sqrt

from config import AppConfig
from utils import Chunk, RetrievalFilters, RetrievedChunk


INDEX_SCHEMA_VERSION = "2026-obsidian-rag-schema-5"
SAVED_ANSWER_DISTANCE_PENALTY = 0.12


class VectorStore:
    """Wrapper around a persistent ChromaDB collection."""

    def __init__(self, config: AppConfig) -> None:
        self.db_path = config.chroma_db_path
        self.client = chromadb.PersistentClient(path=str(config.chroma_db_path))
        self.collection_name = config.chroma_collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.version_file = self.db_path / ".index_schema_version"

    def reset(self) -> None:
        """Delete and recreate the configured collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.write_index_version(INDEX_SCHEMA_VERSION)

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
                    "source_kind": chunk.source_kind,
                    "source_type": chunk.source_type,
                    "content_scope": chunk.content_scope,
                    "content_category": chunk.content_category,
                    "import_genre": chunk.import_genre,
                    "arrangement_track_name": chunk.arrangement_track_name,
                    "arrangement_genre": chunk.arrangement_genre,
                    "arrangement_section_id": chunk.arrangement_section_id,
                    "arrangement_section_name": chunk.arrangement_section_name,
                    "arrangement_energy": (
                        chunk.arrangement_energy if chunk.arrangement_energy is not None else ""
                    ),
                    "arrangement_version": chunk.arrangement_version,
                    "video_title": chunk.video_title,
                    "video_channel_name": chunk.video_channel_name,
                    "video_source_url": chunk.video_source_url,
                    "video_section_title": chunk.video_section_title,
                    "video_start_time": chunk.video_start_time,
                    "video_end_time": chunk.video_end_time,
                    "video_duration_seconds": (
                        chunk.video_duration_seconds if chunk.video_duration_seconds is not None else ""
                    ),
                    "video_language": chunk.video_language,
                    "video_schema_version": chunk.video_schema_version,
                    "video_chunk_kind": chunk.video_chunk_kind,
                    "tags_serialized": _serialize_tags(chunk.tags),
                    "linked_note_keys_serialized": _serialize_values(chunk.linked_note_keys),
                }
                for chunk in chunks
            ],
        )
        self.write_index_version(INDEX_SCHEMA_VERSION)

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
        retrieval_scope: str = "extended",
        include_saved_answers: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Query the vector store and return retrieved chunks."""
        if (filters and (filters.path_contains or filters.tag)) or include_saved_answers is False:
            return self._query_with_post_filters(
                query_embedding,
                top_k,
                filters or RetrievalFilters(),
                retrieval_scope=retrieval_scope,
                include_saved_answers=include_saved_answers,
            )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=self._build_where(filters, retrieval_scope=retrieval_scope),
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
        retrieval_scope: str = "extended",
        include_saved_answers: bool | None = None,
    ) -> list[tuple[str, dict[str, object], list[float]]]:
        """Return all chunk documents, metadata, and embeddings for filtered search."""
        results = self.collection.get(
            include=["documents", "metadatas", "embeddings"],
            where=self._build_where(filters, retrieval_scope=retrieval_scope),
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings", [])
        chunk_rows: list[tuple[str, dict[str, object], list[float]]] = []
        for document, metadata, embedding in zip(documents, metadatas, embeddings):
            if not document or not metadata or embedding is None:
                continue
            if include_saved_answers is False and _is_saved_answer_metadata(metadata):
                continue
            chunk_rows.append((document, dict(metadata), list(embedding)))
        return chunk_rows

    def get_chunks_by_note_keys(
        self,
        note_keys: list[str],
        *,
        max_chunks_per_note: int,
        retrieval_scope: str = "extended",
        excluded_note_keys: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return a small number of chunks for the requested linked notes."""
        linked_chunks: list[RetrievedChunk] = []
        excluded_note_keys = excluded_note_keys or set()

        for note_key in note_keys:
            if note_key in excluded_note_keys:
                continue
            results = self.collection.get(
                include=["documents", "metadatas"],
                where=self._combine_conditions(
                    {"note_key": note_key},
                    self._build_where(None, retrieval_scope=retrieval_scope),
                ),
            )
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            rows = [
                (document, metadata)
                for document, metadata in zip(documents, metadatas)
                if document and metadata
            ]
            rows.sort(key=lambda row: int(row[1].get("chunk_index", 0)))

            for document, metadata in rows[:max_chunks_per_note]:
                linked_metadata = dict(metadata)
                linked_metadata["linked_context"] = True
                linked_chunks.append(
                    RetrievedChunk(
                        text=document,
                        metadata=linked_metadata,
                        distance_or_score=_adjust_distance_for_source_kind(
                            None,
                            linked_metadata,
                        ),
                    )
                )

        return linked_chunks

    def _query_with_post_filters(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: RetrievalFilters,
        *,
        retrieval_scope: str,
        include_saved_answers: bool | None,
    ) -> list[RetrievedChunk]:
        normalized_substring = filters.path_contains.lower() if filters.path_contains else None
        normalized_tag = filters.tag.lower() if filters.tag else None
        candidates = self.get_all_chunks(
            filters=RetrievalFilters(folder=filters.folder),
            retrieval_scope=retrieval_scope,
            include_saved_answers=include_saved_answers,
        )

        filtered_candidates: list[RetrievedChunk] = []
        for document, metadata, embedding in candidates:
            source_path = str(metadata.get("source_path_normalized", "")).lower()
            tags = _deserialize_tags(metadata.get("tags_serialized", ""))
            if normalized_substring and normalized_substring not in source_path:
                continue
            if normalized_tag and normalized_tag not in tags:
                continue
            filtered_candidates.append(
                    RetrievedChunk(
                        text=document,
                        metadata=metadata,
                        distance_or_score=_adjust_distance_for_source_kind(
                            _cosine_distance(query_embedding, embedding),
                            metadata,
                        ),
                    )
                )

        filtered_candidates.sort(
            key=lambda chunk: float("inf")
            if chunk.distance_or_score is None
            else chunk.distance_or_score
        )
        return filtered_candidates[:top_k]

    def _build_where(
        self,
        filters: RetrievalFilters | None,
        *,
        retrieval_scope: str,
    ) -> dict[str, object] | None:
        conditions: list[dict[str, object]] = []
        if filters and filters.folder:
            conditions.append({"source_dir": filters.folder})
        if retrieval_scope == "knowledge":
            conditions.append({"content_scope": "knowledge"})
        return self._combine_conditions(*conditions)

    def _combine_conditions(self, *conditions: dict[str, object] | None) -> dict[str, object] | None:
        filtered = [condition for condition in conditions if condition]
        if not filtered:
            return None
        if len(filtered) == 1:
            return filtered[0]
        return {"$and": filtered}

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
                        distance_or_score=_adjust_distance_for_source_kind(distance, metadata),
                    )
                )
        return retrieved

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def is_index_compatible(self) -> bool:
        """Return whether the stored index matches the current schema version."""
        if self.count() == 0:
            return True
        return self.read_index_version() == INDEX_SCHEMA_VERSION

    def read_index_version(self) -> str:
        """Read the stored local index schema version."""
        if not self.version_file.exists():
            return ""
        return self.version_file.read_text(encoding="utf-8").strip()

    def write_index_version(self, version: str) -> None:
        """Persist the local index schema version."""
        self.version_file.write_text(version, encoding="utf-8")


def _cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = numerator / (left_norm * right_norm)
    return 1 - similarity


def _serialize_tags(tags: tuple[str, ...]) -> str:
    return _serialize_values(tags)


def _adjust_distance_for_source_kind(
    distance: float | None,
    metadata: dict[str, object],
) -> float | None:
    if distance is None:
        return None
    if str(metadata.get("source_kind", "")).strip().lower() == "saved_answer":
        return distance + SAVED_ANSWER_DISTANCE_PENALTY
    return distance


def _is_saved_answer_metadata(metadata: dict[str, object]) -> bool:
    return str(metadata.get("source_kind", "")).strip().lower() == "saved_answer"


def _deserialize_tags(value: object) -> tuple[str, ...]:
    return _deserialize_values(value)


def _serialize_values(values: tuple[str, ...]) -> str:
    return "|".join(values)


def _deserialize_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        return ()
    return tuple(part for part in value.split("|") if part)
