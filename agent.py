"""High-level orchestration for question answering."""

from __future__ import annotations

from llm import OllamaChatClient
from retriever import Retriever
from utils import AnswerResult


class ResearchAgent:
    """Retrieve note context and generate a grounded answer."""

    def __init__(self, retriever: Retriever, chat_client: OllamaChatClient) -> None:
        self.retriever = retriever
        self.chat_client = chat_client

    def answer(self, question: str) -> AnswerResult:
        """Answer a question from the indexed Obsidian vault."""
        chunks = self.retriever.retrieve(question)
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
