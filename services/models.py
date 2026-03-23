"""Structured service-layer results for CLI and UI consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from utils import AnswerResult, RetrievedChunk, RetrievalFilters, RetrievalOptions
from web_search import WebSearchResult


@dataclass(slots=True)
class QueryRequest:
    """Structured request for answering a question."""

    question: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    auto_save: bool = False
    save_title: str | None = None
    retrieval_mode: str = "local_only"


@dataclass(slots=True)
class QueryDebugInfo:
    """Structured retrieval debug information for UI and diagnostics."""

    initial_candidates: list[RetrievedChunk] = field(default_factory=list)
    primary_chunks: list[RetrievedChunk] = field(default_factory=list)
    reranking_applied: bool = False
    reranking_changed: bool = False
    retrieval_filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    retrieval_options: RetrievalOptions = field(default_factory=RetrievalOptions)
    retrieval_mode_requested: str = "local_only"
    retrieval_mode_used: str = "local_only"
    local_retrieval_weak: bool = False
    web_used: bool = False


@dataclass(slots=True)
class QueryResponse:
    """Structured response for query operations."""

    answer_result: AnswerResult
    warnings: list[str] = field(default_factory=list)
    linked_context_chunks: list[RetrievedChunk] = field(default_factory=list)
    web_results: list[WebSearchResult] = field(default_factory=list)
    saved_path: Path | None = None
    debug: QueryDebugInfo = field(default_factory=QueryDebugInfo)

    @property
    def answer(self) -> str:
        return self.answer_result.answer

    @property
    def sources(self) -> list[str]:
        return self.answer_result.sources

    @property
    def retrieved_chunks(self) -> list[RetrievedChunk]:
        return self.answer_result.retrieved_chunks

    @property
    def has_saved(self) -> bool:
        return self.saved_path is not None

    @property
    def web_used(self) -> bool:
        return bool(self.web_results)

    @property
    def local_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Local]")]

    @property
    def web_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Web]")]


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
    vault_path: Path | None = None
    output_path: Path | None = None
    chat_model: str = ""
    embedding_model: str = ""
    ollama_reachable: bool | None = None
    ollama_status_message: str = ""
    ready: bool = False
    index_version: str = ""
