"""Structured service-layer results for CLI and UI consumers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from utils import AnswerResult, RetrievedChunk, RetrievalFilters, RetrievalOptions
from web_search import WebSearchResult


class RetrievalMode(StrEnum):
    """Supported retrieval modes for local and web evidence."""

    LOCAL_ONLY = "local_only"
    AUTO = "auto"
    HYBRID = "hybrid"

    @classmethod
    def coerce(cls, value: "RetrievalMode | str | None") -> "RetrievalMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.LOCAL_ONLY
        normalized = str(value).strip().lower()
        for mode in cls:
            if mode.value == normalized:
                return mode
        raise ValueError(
            f"Unsupported retrieval mode: {value}. Expected one of: "
            f"{', '.join(mode.value for mode in cls)}."
        )


class RetrievalModeUsed(StrEnum):
    """Resolved retrieval behavior used for the final answer."""

    LOCAL_ONLY = "local_only"
    AUTO_LOCAL_ONLY = "auto_local_only"
    AUTO_WITH_WEB = "auto_with_web"
    HYBRID = "hybrid"
    HYBRID_NO_WEB_RESULTS = "hybrid_no_web_results"


class WebQueryStrategy(StrEnum):
    """How the external web query was derived."""

    RAW_QUESTION = "raw_question"
    LOCAL_GUIDED = "local_guided"


@dataclass(slots=True)
class WebSearchAttemptInfo:
    """A single external web-search attempt and its outcome."""

    query: str
    strategy: WebQueryStrategy
    retry_used: bool = False
    provider_returned_results: bool = False
    provider_result_count: int = 0
    usable_result_count: int = 0
    filtered_count: int = 0
    results_discarded_by_filter: bool = False
    failure_reason: str = ""
    outcome: str = "not_attempted"


class AnswerMode(StrEnum):
    """Supported answer-generation modes."""

    STRICT = "strict"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"

    @classmethod
    def coerce(cls, value: "AnswerMode | str | None") -> "AnswerMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.BALANCED
        normalized = str(value).strip().lower()
        for mode in cls:
            if mode.value == normalized:
                return mode
        raise ValueError(
            f"Unsupported answer mode: {value}. Expected one of: "
            f"{', '.join(mode.value for mode in cls)}."
        )


class WorkflowMode(StrEnum):
    """Supported top-level ask workflows."""

    DIRECT = "direct"
    RESEARCH = "research"


@dataclass(slots=True)
class QueryRequest:
    """Structured request for answering a question."""

    question: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    auto_save: bool = False
    save_title: str | None = None
    retrieval_mode: RetrievalMode = RetrievalMode.LOCAL_ONLY
    answer_mode: AnswerMode = AnswerMode.BALANCED

    def __post_init__(self) -> None:
        self.retrieval_mode = RetrievalMode.coerce(self.retrieval_mode)
        self.answer_mode = AnswerMode.coerce(self.answer_mode)


@dataclass(slots=True)
class QueryDebugInfo:
    """Structured retrieval debug information for UI and diagnostics."""

    initial_candidates: list[RetrievedChunk] = field(default_factory=list)
    primary_chunks: list[RetrievedChunk] = field(default_factory=list)
    reranking_applied: bool = False
    reranking_changed: bool = False
    retrieval_filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    retrieval_options: RetrievalOptions = field(default_factory=RetrievalOptions)
    retrieval_mode_requested: RetrievalMode = RetrievalMode.LOCAL_ONLY
    retrieval_mode_used: RetrievalModeUsed = RetrievalModeUsed.LOCAL_ONLY
    answer_mode_requested: AnswerMode = AnswerMode.BALANCED
    answer_mode_used: AnswerMode = AnswerMode.BALANCED
    local_retrieval_weak: bool = False
    web_used: bool = False
    evidence_types_used: tuple[str, ...] = ()
    inference_used: bool = False
    citation_labels: tuple[str, ...] = ()
    hallucination_guard_warnings: tuple[str, ...] = ()
    web_query_used: str = ""
    web_query_strategy: WebQueryStrategy = WebQueryStrategy.RAW_QUESTION
    web_results_filtered_count: int = 0
    web_alignment_warning: str = ""
    web_attempts: list[WebSearchAttemptInfo] = field(default_factory=list)
    web_failure_reason: str = ""
    web_provider_returned_results: bool = False
    web_results_discarded_by_filter: bool = False
    web_retry_used: bool = False


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
    def answer_mode_used(self) -> AnswerMode:
        return self.debug.answer_mode_used

    @property
    def inference_used(self) -> bool:
        return self.debug.inference_used

    @property
    def evidence_types_used(self) -> tuple[str, ...]:
        return self.debug.evidence_types_used

    @property
    def local_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Local")]

    @property
    def saved_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Saved")]

    @property
    def web_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Web")]

    def with_saved_path(self, saved_path: Path) -> "QueryResponse":
        """Return a copy with the saved path filled in while preserving evidence state."""
        return replace(self, saved_path=saved_path)


@dataclass(slots=True)
class ResearchRequest:
    """Structured request for a multi-step research workflow."""

    goal: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    retrieval_mode: RetrievalMode = RetrievalMode.LOCAL_ONLY
    answer_mode: AnswerMode = AnswerMode.BALANCED
    max_subquestions: int = 3
    auto_save: bool = False
    save_title: str | None = None

    def __post_init__(self) -> None:
        self.retrieval_mode = RetrievalMode.coerce(self.retrieval_mode)
        self.answer_mode = AnswerMode.coerce(self.answer_mode)
        self.max_subquestions = max(1, min(int(self.max_subquestions), 5))


@dataclass(slots=True)
class ResearchStepResult:
    """A single explicit research step with its answer output."""

    subquestion: str
    response: QueryResponse
    completed: bool = True


@dataclass(slots=True)
class ResearchResponse:
    """Structured response for research workflow operations."""

    goal: str
    subquestions: list[str]
    steps: list[ResearchStepResult]
    answer_result: AnswerResult
    warnings: list[str] = field(default_factory=list)
    saved_path: Path | None = None
    planning_notes: list[str] = field(default_factory=list)

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
    def web_results(self) -> list[WebSearchResult]:
        web_results: list[WebSearchResult] = []
        seen: set[tuple[str, str]] = set()
        for step in self.steps:
            for result in step.response.web_results:
                key = (result.title, result.url)
                if key in seen:
                    continue
                seen.add(key)
                web_results.append(result)
        return web_results

    @property
    def has_saved(self) -> bool:
        return self.saved_path is not None

    @property
    def local_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Local")]

    @property
    def saved_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Saved")]

    @property
    def web_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Web")]

    @property
    def inference_used(self) -> bool:
        return "[Inference]" in self.answer_result.answer

    def with_saved_path(self, saved_path: Path) -> "ResearchResponse":
        """Return a copy with the saved path filled in."""
        return replace(self, saved_path=saved_path)


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


@dataclass(slots=True)
class IngestionRequest:
    """Structured request for importing an external source into the vault."""

    source: str
    title_override: str | None = None
    index_now: bool | None = None


@dataclass(slots=True)
class IngestionResponse:
    """Structured response for an ingestion workflow."""

    source: str
    source_type: str
    saved_path: Path
    title: str
    index_triggered: bool = False
    warnings: list[str] = field(default_factory=list)
