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
    # Core identity
    genre: str | None = None
    bpm: int | None = None
    key: str | None = None
    vibe: list[str] = field(default_factory=list)
    # References
    reference_tracks: list[str] = field(default_factory=list)
    # Current working state
    current_stage: str | None = None
    current_problem: str | None = None
    # Persistent issues + goals
    known_issues: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    # Optional section-aware track memory
    sections: dict[str, "SectionContext"] = field(default_factory=dict)


@dataclass(slots=True)
class SectionContext:
    """Lightweight section-specific track context for arrangement-aware collaboration."""

    name: str
    bars: str = ""
    role: str = ""
    energy_level: str = ""
    elements: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class TrackContextUpdateProposal:
    """Structured, reviewable assistant proposal for updating active Track Context."""

    track_id: str = ""
    summary: str = ""
    set_fields: dict[str, object] = field(default_factory=dict)
    add_to_lists: dict[str, list[str]] = field(default_factory=dict)
    remove_from_lists: dict[str, list[str]] = field(default_factory=dict)
    set_sections: dict[str, dict[str, object]] = field(default_factory=dict)
    add_section_issues: dict[str, list[str]] = field(default_factory=dict)
    remove_section_issues: dict[str, list[str]] = field(default_factory=dict)
    add_section_elements: dict[str, list[str]] = field(default_factory=dict)
    add_section_notes: dict[str, list[str]] = field(default_factory=dict)
    section_focus: str = ""
    confidence: str = ""
    source_reasoning: str = ""

    def is_empty(self) -> bool:
        """Return whether the proposal contains any actual Track Context changes."""
        return not (
            self.set_fields
            or self.add_to_lists
            or self.remove_from_lists
            or self.set_sections
            or self.add_section_issues
            or self.remove_section_issues
            or self.add_section_elements
            or self.add_section_notes
            or bool(self.section_focus.strip())
        )


@dataclass(slots=True)
class ArrangementSectionIndexEntry:
    """A lightweight row from the arrangement section index."""

    id: str
    name: str
    bars: str | None = None
    start_bar: int | None = None
    end_bar: int | None = None
    energy: int | None = None
    themes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ArrangementSection:
    """A parsed arrangement section with timeline, layers, and section-specific guidance."""

    id: str
    name: str
    start_bar: int | None = None
    end_bar: int | None = None
    energy: int | None = None
    elements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    purpose: str | None = None


@dataclass(slots=True)
class ArrangementDocument:
    """A parsed track arrangement document focused on structural track arrangement."""

    track_id: str | None = None
    track_name: str | None = None
    total_bars: int | None = None
    genre: str | None = None
    bpm: int | None = None
    key: str | None = None
    status: str | None = None
    reference_tracks: list[str] = field(default_factory=list)
    arrangement_version: int | str | None = None
    global_notes: list[str] = field(default_factory=list)
    section_index: list[ArrangementSectionIndexEntry] = field(default_factory=list)
    sections: list[ArrangementSection] = field(default_factory=list)


@dataclass(slots=True)
class VideoTranscriptSegment:
    """A timestamped transcript segment extracted from a video source."""

    text: str
    start_time: float
    end_time: float


@dataclass(slots=True)
class VideoKnowledgeSection:
    """A semantic video section prepared for note rendering and retrieval."""

    title: str
    start_time: float
    end_time: float
    summary: str
    key_points: list[str] = field(default_factory=list)
    content: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoKnowledgeDocument:
    """Structured representation of a video-derived knowledge note."""

    source_type: str = "youtube_video"
    source_url: str = ""
    video_title: str = ""
    platform: str = "youtube"
    channel_name: str | None = None
    published_at: str | None = None
    duration_seconds: int | None = None
    duration_readable: str | None = None
    language: str | None = None
    imported_at: str | None = None
    schema_version: str = "video_import_v1"
    content_type: str = "video_knowledge"
    status: str = "imported"
    indexed: bool = False
    created_by: str = "obsidian_track_collaborator"
    video_id: str | None = None
    transcript_source: str | None = None
    whisper_model: str | None = None
    video_index_mode: str = "sections"
    description_present: bool | None = None
    thumbnail_url: str | None = None
    retrieval_ready: bool = True
    section_count: int = 0
    transcript_chunk_count: int = 0
    domain_profile: str | None = None
    workflow_type: str | None = None
    import_notes: str | None = None
    import_genre: str | None = None
    knowledge_category: str | None = None
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    sections: list[VideoKnowledgeSection] = field(default_factory=list)
    producer_notes: list[str] = field(default_factory=list)
    retrieval_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryRequest:
    """Structured request for answering a question."""

    question: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    auto_save: bool = False
    save_title: str | None = None
    chat_provider_override: str | None = None
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
    section_focus: str | None = None
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
    priority: str = "medium"
    linked_section: str = ""
    completed_at: str | None = None


@dataclass(slots=True)
class PersistedTrackTask:
    """A canonical persisted task item tied to a YAML Track Context track."""

    id: str
    text: str
    status: str
    priority: str
    linked_section: str
    created_from: str
    created_at: str
    completed_at: str | None = None
    notes: str = ""


@dataclass(slots=True)
class RetrievalScoreDebug:
    """Structured debug row for a reranked retrieval candidate."""

    note_title: str
    source_path: str
    chunk_index: int
    final_score: float
    component_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class QueryDebugInfo:
    """Structured retrieval debug information for UI and diagnostics."""

    initial_candidates: list[RetrievedChunk] = field(default_factory=list)
    primary_chunks: list[RetrievedChunk] = field(default_factory=list)
    reranking_applied: bool = False
    reranking_changed: bool = False
    reranking_details: list[RetrievalScoreDebug] = field(default_factory=list)
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
    active_chat_provider: str = ""
    active_chat_model: str = ""
    imported_genres_eligible: tuple[str, ...] = ()
    response_mode_selected: str = "direct_answer"
    followup_triggered: bool = False
    missing_dimension: str = ""
    active_section: str = ""
    loaded_task_count: int = 0
    open_task_count: int = 0
    active_task_summaries: tuple[str, ...] = ()


@dataclass(slots=True)
class TrackContextSuggestions:
    """Assistant-suggested Track Context updates awaiting user review."""

    known_issues: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    current_stage: str | None = None
    current_problem: str | None = None
    vibe_suggestions: list[str] = field(default_factory=list)
    reference_track_suggestions: list[str] = field(default_factory=list)
    section_suggestions: list[str] = field(default_factory=list)
    section_focus: str | None = None
    bpm_suggestion: int | None = None
    key_suggestion: str | None = None

    def is_empty(self) -> bool:
        """Return whether the suggestion payload contains any actionable updates."""
        return not (
            self.known_issues
            or self.goals
            or self.current_stage
            or self.current_problem
            or self.vibe_suggestions
            or self.reference_track_suggestions
            or self.section_suggestions
            or self.section_focus
            or self.bpm_suggestion is not None
            or self.key_suggestion is not None
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
    track_context_update: TrackContextUpdateProposal | None = None
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

    @property
    def track_context_update_available(self) -> bool:
        return self.track_context_update is not None and not self.track_context_update.is_empty()

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
    chat_provider_override: str | None = None
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
    active_chat_provider: str = ""
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
    import_genre: str | None = None
    knowledge_category: str | None = None


@dataclass(slots=True)
class IngestionResponse:
    """Structured response for an ingestion workflow."""

    source: str
    source_type: str
    saved_path: Path
    title: str
    import_genre: str | None = None
    knowledge_category: str | None = None
    index_triggered: bool = False
    section_count: int = 0
    transcript_chunk_count: int = 0
    warnings: list[str] = field(default_factory=list)
