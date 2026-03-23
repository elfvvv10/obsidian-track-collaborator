"""Thin index/build service built on existing core modules."""

from __future__ import annotations

from config import AppConfig
from chunker import chunk_notes
from embeddings import OllamaEmbeddingClient
from services.common import check_ollama_status, ensure_index_compatible, resolve_note_links
from services.models import IndexResponse
from utils import Chunk, Note, get_logger
from vault_loader import load_notes
from vector_store import VectorStore


logger = get_logger()


class IndexService:
    """Coordinate loading, chunking, and indexing without duplicating core logic."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def index(self, *, reset_store: bool) -> IndexResponse:
        """Build or rebuild the vector index."""
        logger.info("Loading notes from %s", self.config.obsidian_vault_path)
        excluded_paths = []
        if self.config.obsidian_output_path != self.config.obsidian_vault_path:
            excluded_paths.append(self.config.obsidian_output_path)

        notes = load_notes(self.config.obsidian_vault_path, excluded_paths=excluded_paths)
        resolve_note_links(notes)

        embedding_client = OllamaEmbeddingClient(self.config)
        vector_store = VectorStore(self.config)
        if reset_store:
            logger.info("Resetting Chroma collection")
            vector_store.reset()
        else:
            ensure_index_compatible(vector_store)

        existing_fingerprints = {} if reset_store else vector_store.list_note_fingerprints()
        if not notes:
            if existing_fingerprints:
                logger.info("Vault is empty. Removing %s indexed note(s).", len(existing_fingerprints))
                vector_store.delete_by_note_keys(list(existing_fingerprints))
                return IndexResponse(
                    deleted_notes_removed=len(existing_fingerprints),
                    total_chunks_stored=vector_store.count(),
                    reset_performed=reset_store,
                    warnings=["Vault is empty; removed previously indexed notes."],
                    vault_path=self.config.obsidian_vault_path,
                    output_path=self.config.obsidian_output_path,
                    chat_model=self.config.ollama_chat_model,
                    embedding_model=self.config.ollama_embedding_model,
                    index_version=vector_store.read_index_version(),
                    ready=vector_store.count() > 0 and vector_store.is_index_compatible(),
                )
            raise RuntimeError("No markdown notes were found in the configured vault.")

        chunks = chunk_notes(
            notes,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy,
        )
        if not chunks:
            raise RuntimeError("No note chunks were created from the vault contents.")

        logger.info("Loaded %s notes and created %s chunks", len(notes), len(chunks))

        note_chunks = _group_chunks_by_note_key(chunks)
        deleted_note_keys = sorted(set(existing_fingerprints) - set(note_chunks))
        if deleted_note_keys:
            logger.info("Removing %s deleted note(s) from the index", len(deleted_note_keys))
            vector_store.delete_by_note_keys(deleted_note_keys)

        chunks_to_index = _select_chunks_to_index(note_chunks, existing_fingerprints, vector_store)
        if not chunks_to_index:
            logger.info("Index already up to date. Stored %s chunks.", vector_store.count())
            return IndexResponse(
                notes_loaded=len(notes),
                chunks_created=len(chunks),
                deleted_notes_removed=len(deleted_note_keys),
                total_chunks_stored=vector_store.count(),
                reset_performed=reset_store,
                up_to_date=True,
                vault_path=self.config.obsidian_vault_path,
                output_path=self.config.obsidian_output_path,
                chat_model=self.config.ollama_chat_model,
                embedding_model=self.config.ollama_embedding_model,
                index_version=vector_store.read_index_version(),
                ready=vector_store.count() > 0 and vector_store.is_index_compatible(),
            )

        logger.info(
            "Generating embeddings for %s updated chunk(s) with Ollama model '%s'",
            len(chunks_to_index),
            self.config.ollama_embedding_model,
        )
        embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks_to_index])

        logger.info("Writing updated chunks to ChromaDB at %s", self.config.chroma_db_path)
        vector_store.upsert_chunks(chunks_to_index, embeddings)
        logger.info("Index complete. Stored %s chunks.", vector_store.count())
        return IndexResponse(
            notes_loaded=len(notes),
            chunks_created=len(chunks),
            chunks_indexed=len(chunks_to_index),
            deleted_notes_removed=len(deleted_note_keys),
            total_chunks_stored=vector_store.count(),
            reset_performed=reset_store,
            vault_path=self.config.obsidian_vault_path,
            output_path=self.config.obsidian_output_path,
            chat_model=self.config.ollama_chat_model,
            embedding_model=self.config.ollama_embedding_model,
            index_version=vector_store.read_index_version(),
            ready=vector_store.count() > 0 and vector_store.is_index_compatible(),
        )

    def get_status(self) -> IndexResponse:
        """Return lightweight index status for CLI or UI display."""
        vector_store = VectorStore(self.config)
        ollama_reachable, ollama_status_message = check_ollama_status(
            self.config.ollama_base_url,
            timeout_seconds=min(self.config.ollama_timeout_seconds, 5),
        )
        chunk_count = vector_store.count()
        index_compatible = vector_store.is_index_compatible()
        return IndexResponse(
            total_chunks_stored=chunk_count,
            index_compatible=index_compatible,
            vault_path=self.config.obsidian_vault_path,
            output_path=self.config.obsidian_output_path,
            chat_model=self.config.ollama_chat_model,
            embedding_model=self.config.ollama_embedding_model,
            ollama_reachable=ollama_reachable,
            ollama_status_message=ollama_status_message,
            ready=chunk_count > 0 and index_compatible,
            index_version=vector_store.read_index_version(),
        )


def _group_chunks_by_note_key(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    grouped_chunks: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped_chunks.setdefault(chunk.note_key, []).append(chunk)
    return grouped_chunks


def _select_chunks_to_index(
    note_chunks: dict[str, list[Chunk]],
    existing_fingerprints: dict[str, str],
    vector_store: VectorStore,
) -> list[Chunk]:
    chunks_to_index: list[Chunk] = []

    for note_key, note_chunk_group in note_chunks.items():
        current_fingerprint = note_chunk_group[0].note_fingerprint
        existing_fingerprint = existing_fingerprints.get(note_key)
        if current_fingerprint == existing_fingerprint:
            continue

        if existing_fingerprint:
            vector_store.delete_by_note_keys([note_key])
        chunks_to_index.extend(note_chunk_group)

    return chunks_to_index
