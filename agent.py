"""High-level orchestration for question answering."""

from __future__ import annotations

from model_clients import ChatModelClient
from retriever import Retriever
from services.models import RetrievalScope
from utils import AnswerResult, RetrievalFilters, RetrievalOptions


class ResearchAgent:
    """Retrieve note context and generate a grounded answer."""

    def __init__(self, retriever: Retriever, chat_client: ChatModelClient) -> None:
        self.retriever = retriever
        self.chat_client = chat_client

    def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
        options: RetrievalOptions | None = None,
        retrieval_scope: RetrievalScope = RetrievalScope.KNOWLEDGE,
    ) -> AnswerResult:
        """Answer a question from the indexed Obsidian vault."""
        chunks = self.retriever.retrieve(
            question,
            filters=filters,
            options=options,
            retrieval_scope=retrieval_scope,
        )
        if not chunks:
            return AnswerResult(
                answer="No relevant note context matched the current retrieval filters.",
                sources=[],
                retrieved_chunks=[],
            )
        answer = self.chat_client.answer_question(question, chunks)

        seen_sources: set[str] = set()
        sources: list[str] = []
        for chunk in chunks:
            source = f"{chunk.metadata.get('note_title', 'Untitled')} ({chunk.metadata.get('source_path', 'unknown')})"
            if source in seen_sources:
                continue
            seen_sources.add(source)
            sources.append(source)

        return AnswerResult(answer=answer, sources=sources, retrieved_chunks=chunks)
