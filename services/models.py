"""Structured service-layer results for CLI and UI consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from utils import AnswerResult, RetrievedChunk, RetrievalFilters, RetrievalOptions


@dataclass(slots=True)
class QueryRequest:
    """Structured request for answering a question."""

    question: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    auto_save: bool = False


@dataclass(slots=True)
class QueryResponse:
    """Structured response for query operations."""

    answer_result: AnswerResult
    warnings: list[str] = field(default_factory=list)
    linked_context_chunks: list[RetrievedChunk] = field(default_factory=list)
    saved_path: Path | None = None

    @property
    def answer(self) -> str:
        return self.answer_result.answer

    @property
    def sources(self) -> list[str]:
        return self.answer_result.sources

    @property
    def retrieved_chunks(self) -> list[RetrievedChunk]:
        return self.answer_result.retrieved_chunks


@dataclass(slots=True)
class IndexResponse:
    """Structured response for index and rebuild operations."""

    notes_loaded: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    deleted_notes_removed: int = 0
    total_chunks_stored: int = 0
    reset_performed: bool = False
    up_to_date: bool = False
    index_compatible: bool = True
    warnings: list[str] = field(default_factory=list)
