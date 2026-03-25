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


class RetrievalScope(StrEnum):
    """Supported local retrieval scopes."""

    KNOWLEDGE = "knowledge"
    EXTENDED = "extended"

    @classmethod
    def coerce(cls, value: "RetrievalScope | str | None") -> "RetrievalScope":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.KNOWLEDGE
        normalized = str(value).strip().lower()
        for scope in cls:
            if scope.value == normalized:
                return scope
        raise ValueError(
            f"Unsupported retrieval scope: {value}. Expected one of: "
            f"{', '.join(scope.value for scope in cls)}."
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


class DomainProfile(StrEnum):
    """Supported assistant domain profiles."""

    ELECTRONIC_MUSIC = "electronic_music"

    @classmethod
    def coerce(cls, value: "DomainProfile | str | None") -> "DomainProfile":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.ELECTRONIC_MUSIC
        normalized = str(value).strip().lower()
        for profile in cls:
            if profile.value == normalized:
                return profile
        raise ValueError(
            f"Unsupported domain profile: {value}. Expected one of: "
            f"{', '.join(profile.value for profile in cls)}."
        )


class CollaborationWorkflow(StrEnum):
    """Supported music collaboration workflows."""

    GENERAL_ASK = "general_ask"
    GENRE_FIT_REVIEW = "genre_fit_review"
    TRACK_CONCEPT_CRITIQUE = "track_concept_critique"
    ARRANGEMENT_PLANNER = "arrangement_planner"
    SOUND_DESIGN_BRAINSTORM = "sound_design_brainstorm"
    RESEARCH_SESSION = "research_session"

    @classmethod
    def coerce(cls, value: "CollaborationWorkflow | str | None") -> "CollaborationWorkflow":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.GENERAL_ASK
        normalized = str(value).strip().lower()
        for workflow in cls:
            if workflow.value == normalized:
                return workflow
        raise ValueError(
            f"Unsupported collaboration workflow: {value}. Expected one of: "
            f"{', '.join(workflow.value for workflow in cls)}."
        )


@dataclass(slots=True)
class WorkflowInput:
    """Optional structured fields that support music collaboration workflows."""

    genre: str | None = None
    bpm: str | None = None
    references: str | None = None
    mood: str | None = None
    arrangement_notes: str | None = None
    instrumentation: str | None = None
    sound_palette: str | None = None
    energy_goal: str | None = None
    track_length: str | None = None
    role_of_key_elements: str | None = None
    track_context_path: str | None = None

    def as_dict(self) -> dict[str, str]:
        """Return only filled workflow input fields."""
        return {
            key: value.strip()
            for key, value in {
                "genre": self.genre,
                "bpm": self.bpm,
                "references": self.references,
                "mood": self.mood,
                "arrangement_notes": self.arrangement_notes,
                "instrumentation": self.instrumentation,
                "sound_palette": self.sound_palette,
                "energy_goal": self.energy_goal,
                "track_length": self.track_length,
                "role_of_key_elements": self.role_of_key_elements,
                "track_context_path": self.track_context_path,
            }.items()
            if value and value.strip()
        }


@dataclass(slots=True)
class TrackContext:
    """Normalized YAML-backed track context used by the primary editable flow."""

    track_id: str = "default_track"
    track_name: str | None = None
    genre: str | None = None
    bpm: int | None = None
    key: str | None = None
    vibe: list[str] = field(default_factory=list)
    reference_tracks: list[str] = field(default_factory=list)
    workflow_mode: str = "general"
    current_stage: str | None = None
    current_section: str | None = None
    sections: dict[str, str] = field(default_factory=dict)
    known_issues: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryRequest:
    """Structured request for answering a question."""

    question: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    auto_save: bool = False
    save_title: str | None = None
    chat_model_override: str | None = None
    retrieval_scope: RetrievalScope = RetrievalScope.KNOWLEDGE
    retrieval_mode: RetrievalMode = RetrievalMode.AUTO
    answer_mode: AnswerMode = AnswerMode.BALANCED
    domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC
    collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.GENERAL_ASK
    workflow_input: WorkflowInput = field(default_factory=WorkflowInput)
    track_id: str | None = None
    use_track_context: bool = True
    track_context: "TrackContext | None" = None
    recent_conversation: list["ChatMessage"] = field(default_factory=list)
    current_tasks: list["SessionTask"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.retrieval_scope = RetrievalScope.coerce(self.retrieval_scope)
        self.retrieval_mode = RetrievalMode.coerce(self.retrieval_mode)
        self.answer_mode = AnswerMode.coerce(self.answer_mode)
        self.domain_profile = DomainProfile.coerce(self.domain_profile)
        self.collaboration_workflow = CollaborationWorkflow.coerce(self.collaboration_workflow)


@dataclass(slots=True)
class ChatMessage:
    """A single in-session chat message used as internal prompt context."""

    role: str
    content: str
    created_at: str


@dataclass(slots=True)
class SessionTask:
    """A lightweight in-session task item used as internal prompt context."""

    id: str
    text: str
    status: str
    source: str
    created_at: str
    notes: str = ""


@dataclass(slots=True)
class QueryDebugInfo:
    """Structured retrieval debug information for UI and diagnostics."""

    initial_candidates: list[RetrievedChunk] = field(default_factory=list)
    primary_chunks: list[RetrievedChunk] = field(default_factory=list)
    reranking_applied: bool = False
    reranking_changed: bool = False
    retrieval_filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    retrieval_options: RetrievalOptions = field(default_factory=RetrievalOptions)
    retrieval_scope_requested: RetrievalScope = RetrievalScope.KNOWLEDGE
    retrieval_mode_requested: RetrievalMode = RetrievalMode.LOCAL_ONLY
    retrieval_mode_used: RetrievalModeUsed = RetrievalModeUsed.LOCAL_ONLY
    answer_mode_requested: AnswerMode = AnswerMode.BALANCED
    answer_mode_used: AnswerMode = AnswerMode.BALANCED
    rewritten_query: str = ""
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
    curated_knowledge_chunks: int = 0
    imported_knowledge_chunks: int = 0
    non_curated_note_chunks: int = 0
    generated_or_imported_chunks: int = 0
    active_chat_model: str = ""


@dataclass(slots=True)
class TrackContextSuggestions:
    """Assistant-suggested Track Context updates awaiting user review."""

    known_issues: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    current_stage: str | None = None
    current_section: str | None = None

    def is_empty(self) -> bool:
        """Return whether the suggestion payload contains any actionable updates."""
        return not (
            self.known_issues
            or self.goals
            or self.notes
            or self.current_stage
            or self.current_section
        )


@dataclass(slots=True)
class QueryResponse:
    """Structured response for query operations."""

    answer_result: AnswerResult
    warnings: list[str] = field(default_factory=list)
    linked_context_chunks: list[RetrievedChunk] = field(default_factory=list)
    web_results: list[WebSearchResult] = field(default_factory=list)
    saved_path: Path | None = None
    debug: QueryDebugInfo = field(default_factory=QueryDebugInfo)
    domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC
    collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.GENERAL_ASK
    workflow_input: WorkflowInput = field(default_factory=WorkflowInput)
    track_context: TrackContext | None = None
    track_context_suggestions: TrackContextSuggestions | None = None

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
    def imported_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Import")]

    @property
    def web_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Web")]

    @property
    def curated_chunks(self) -> list[RetrievedChunk]:
        return [chunk for chunk in self.retrieved_chunks if chunk.metadata.get("content_category") == "curated_knowledge"]

    @property
    def imported_chunks(self) -> list[RetrievedChunk]:
        return [chunk for chunk in self.retrieved_chunks if chunk.metadata.get("content_category") == "imported_knowledge"]

    @property
    def non_curated_chunks(self) -> list[RetrievedChunk]:
        return [chunk for chunk in self.retrieved_chunks if chunk.metadata.get("content_category") == "non_curated_note"]

    @property
    def generated_or_imported_chunks(self) -> list[RetrievedChunk]:
        return [
            chunk
            for chunk in self.retrieved_chunks
            if chunk.metadata.get("content_category") == "generated_or_imported"
        ]

    def with_saved_path(self, saved_path: Path) -> "QueryResponse":
        """Return a copy with the saved path filled in while preserving evidence state."""
        return replace(self, saved_path=saved_path)


@dataclass(slots=True)
class ResearchRequest:
    """Structured request for a multi-step research workflow."""

    goal: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    retrieval_scope: RetrievalScope = RetrievalScope.KNOWLEDGE
    retrieval_mode: RetrievalMode = RetrievalMode.AUTO
    answer_mode: AnswerMode = AnswerMode.BALANCED
    max_subquestions: int = 3
    auto_save: bool = False
    save_title: str | None = None
    chat_model_override: str | None = None
    domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC
    collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.RESEARCH_SESSION
    workflow_input: WorkflowInput = field(default_factory=WorkflowInput)
    track_id: str | None = None
    use_track_context: bool = True
    track_context: TrackContext | None = None

    def __post_init__(self) -> None:
        self.retrieval_scope = RetrievalScope.coerce(self.retrieval_scope)
        self.retrieval_mode = RetrievalMode.coerce(self.retrieval_mode)
        self.answer_mode = AnswerMode.coerce(self.answer_mode)
        self.max_subquestions = max(1, min(int(self.max_subquestions), 5))
        self.domain_profile = DomainProfile.coerce(self.domain_profile)
        self.collaboration_workflow = CollaborationWorkflow.coerce(self.collaboration_workflow)


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
    active_chat_model: str = ""
    domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC
    collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.RESEARCH_SESSION
    workflow_input: WorkflowInput = field(default_factory=WorkflowInput)
    track_context: TrackContext | None = None

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
    def imported_sources(self) -> list[str]:
        return [source for source in self.answer_result.sources if source.startswith("[Import")]

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
