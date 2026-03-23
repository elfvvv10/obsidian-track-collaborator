"""Thin query service built on existing retrieval and answer modules."""

from __future__ import annotations

from config import AppConfig
from agent import ResearchAgent
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from retriever import Retriever
from saver import save_answer
from services.common import ensure_index_compatible
from services.models import QueryRequest, QueryResponse
from utils import AnswerResult, RetrievedChunk, get_logger
from vector_store import VectorStore


logger = get_logger()


class QueryService:
    """Coordinate retrieval, answer generation, and optional save-back."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def ask(self, request: QueryRequest) -> QueryResponse:
        """Run the full question-answer flow and return structured UI-friendly results."""
        vector_store = VectorStore(self.config)
        ensure_index_compatible(vector_store)

        embedding_client = OllamaEmbeddingClient(self.config)
        retriever = Retriever(self.config, embedding_client, vector_store)
        chat_client = OllamaChatClient(self.config)
        agent = ResearchAgent(retriever, chat_client)

        logger.info("Retrieving relevant notes")
        answer_result = agent.answer(
            request.question,
            filters=request.filters,
            options=request.options,
        )

        warnings = _build_warnings(answer_result)
        linked_chunks = [chunk for chunk in answer_result.retrieved_chunks if chunk.metadata.get("linked_context")]
        saved_path = None

        if request.auto_save or self.config.auto_save_answer:
            saved_path = save_answer(self.config.obsidian_output_path, request.question, answer_result)
            logger.info("Saved answer to %s", saved_path)

        return QueryResponse(
            answer_result=answer_result,
            warnings=warnings,
            linked_context_chunks=linked_chunks,
            saved_path=saved_path,
        )

    def save(self, question: str, answer_result: AnswerResult) -> QueryResponse:
        """Persist an existing answer result and return updated response info."""
        saved_path = save_answer(self.config.obsidian_output_path, question, answer_result)
        logger.info("Saved answer to %s", saved_path)
        return QueryResponse(
            answer_result=answer_result,
            warnings=_build_warnings(answer_result),
            linked_context_chunks=[
                chunk for chunk in answer_result.retrieved_chunks if chunk.metadata.get("linked_context")
            ],
            saved_path=saved_path,
        )


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
