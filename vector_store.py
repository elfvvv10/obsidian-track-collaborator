"""ChromaDB persistence helpers."""

from __future__ import annotations

import chromadb

from config import AppConfig
from utils import Chunk, RetrievedChunk


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
                    "note_title": chunk.note_title,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Query the vector store and return retrieved chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

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
