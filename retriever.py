"""Retrieve relevant note chunks for a user query."""

from __future__ import annotations

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from utils import RetrievalFilters, RetrievedChunk
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
    ) -> list[RetrievedChunk]:
        """Return the top-k relevant chunks for a question."""
        if self.vector_store.count() == 0:
            raise RuntimeError("The vector store is empty. Run `python main.py index` first.")

        query_embedding = self.embedding_client.embed_text(query)
        return self.vector_store.query(query_embedding, self.config.top_k_results, filters=filters)
