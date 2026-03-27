"""Weighted local reranking helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from services.models import CollaborationWorkflow, DomainProfile, RetrievalScoreDebug, TrackContext
from utils import RetrievedChunk


_WEIGHTS = {
    "semantic_similarity": 3.0,
    "lexical_overlap": 2.25,
    "title_context_overlap": 1.5,
    "genre_match": 1.8,
    "domain_match": 0.75,
    "importance": 1.0,
    "track_context_relevance": 2.0,
    "workflow_relevance": 0.8,
}

_ELECTRONIC_MUSIC_TYPES = {"track_arrangement", "youtube_video", "webpage_import"}

_IMPORTANCE_BY_CATEGORY = {
    "curated_knowledge": 1.0,
    "imported_knowledge": 0.55,
    "non_curated_note": 0.15,
    "generated_or_imported": -0.2,
}

_IMPORTANCE_BY_SOURCE_KIND = {
    "primary_note": 0.2,
    "saved_answer": -0.4,
}

_WORKFLOW_SOURCE_TYPE_BONUSES = {
    CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE: {"track_arrangement": 1.0},
    CollaborationWorkflow.ARRANGEMENT_PLANNER: {"track_arrangement": 1.0},
    CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM: {"youtube_video": 0.5},
    CollaborationWorkflow.GENRE_FIT_REVIEW: {"youtube_video": 0.35, "webpage_import": 0.25},
}


@dataclass(slots=True)
class _CandidateScore:
    chunk: RetrievedChunk
    final_score: float
    component_scores: dict[str, float]


@dataclass(slots=True)
class _RerankResult:
    """Compatibility wrapper for old list-style and new tuple-unpacking callers."""

    chunks: list[RetrievedChunk]
    details: list[RetrievalScoreDebug]

    def __iter__(self):
        yield self.chunks
        yield self.details

    def __getitem__(self, index):
        return self.chunks[index]

    def __len__(self) -> int:
        return len(self.chunks)


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    boost_tags: tuple[str, ...] = (),
    tag_boost_weight: float = 3.0,
    track_context: TrackContext | None = None,
    collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.GENERAL_ASK,
    section_focus: str | None = None,
    domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC,
) -> tuple[list[RetrievedChunk], list[RetrievalScoreDebug]]:
    """Rerank retrieved chunks with explicit weighted scoring."""
    if not chunks:
        return _RerankResult([], [])

    normalized_query = " ".join(_tokenize(query))
    query_terms = set(_tokenize(query))
    normalized_boost_tags = {tag.lower() for tag in boost_tags if tag}
    scoring_context_terms = _track_context_terms(track_context, section_focus=section_focus or "")

    scored_candidates = []
    for chunk in chunks:
        candidate = _score_candidate(
            chunk,
            query_terms=query_terms,
            normalized_boost_tags=normalized_boost_tags,
            tag_boost_weight=tag_boost_weight,
            track_context=track_context,
            track_context_terms=scoring_context_terms,
            collaboration_workflow=collaboration_workflow,
            section_focus=section_focus or "",
            domain_profile=domain_profile,
        )
        title_text = str(chunk.metadata.get("note_title", ""))
        normalized_title = " ".join(_tokenize(title_text))
        title_terms = set(_tokenize(title_text))

        exact_title_match_boost = 0.0
        if normalized_query and normalized_title and normalized_query in normalized_title:
            exact_title_match_boost = 3.0

        title_token_overlap_boost = 0.0
        if query_terms and title_terms:
            title_token_overlap_boost = (len(query_terms & title_terms) / len(query_terms)) * 1.5

        candidate.component_scores["title_exact_match"] = exact_title_match_boost
        candidate.component_scores["title_token_overlap_boost"] = title_token_overlap_boost
        candidate.final_score += exact_title_match_boost + title_token_overlap_boost
        scored_candidates.append(candidate)

    scored_candidates.sort(
        key=lambda candidate: (
            candidate.final_score,
            -(candidate.chunk.distance_or_score if candidate.chunk.distance_or_score is not None else 1.0),
        ),
        reverse=True,
    )
    reranked_chunks = [candidate.chunk for candidate in scored_candidates]
    reranking_details = [
        RetrievalScoreDebug(
            note_title=str(candidate.chunk.metadata.get("note_title", "Untitled")),
            source_path=str(candidate.chunk.metadata.get("source_path", "")),
            chunk_index=int(candidate.chunk.metadata.get("chunk_index", 0) or 0),
            final_score=round(candidate.final_score, 4),
            component_scores={key: round(value, 4) for key, value in candidate.component_scores.items()},
        )
        for candidate in scored_candidates
    ]
    return _RerankResult(
        reranked_chunks,
        reranking_details,
    )


def _score_candidate(
    chunk: RetrievedChunk,
    *,
    query_terms: set[str],
    normalized_boost_tags: set[str],
    tag_boost_weight: float,
    track_context: TrackContext | None,
    track_context_terms: set[str],
    collaboration_workflow: CollaborationWorkflow,
    section_focus: str,
    domain_profile: DomainProfile,
) -> _CandidateScore:
    chunk_terms = set(_tokenize(chunk.text))
    title_terms = set(_title_terms(chunk))
    context_terms = set(_context_terms(chunk))
    metadata_terms = set(_metadata_terms(chunk))
    searchable_terms = chunk_terms | title_terms | context_terms | metadata_terms

    similarity = max(0.0, 1.0 - (chunk.distance_or_score if chunk.distance_or_score is not None else 1.0))
    lexical_overlap = _overlap_ratio(query_terms, chunk_terms)
    title_context_overlap = _overlap_ratio(query_terms, title_terms | context_terms)
    genre_match = _genre_match_score(track_context, chunk, searchable_terms)
    domain_match = _domain_match_score(domain_profile, chunk)
    importance = _importance_score(chunk)
    track_context_relevance = _track_context_relevance_score(track_context_terms, searchable_terms, chunk, section_focus)
    workflow_relevance = _workflow_relevance_score(collaboration_workflow, chunk)
    tag_boost = len(normalized_boost_tags & set(_metadata_tags(chunk))) * tag_boost_weight

    component_scores = {
        "semantic_similarity": similarity * _WEIGHTS["semantic_similarity"],
        "lexical_overlap": lexical_overlap * _WEIGHTS["lexical_overlap"],
        "title_context_overlap": title_context_overlap * _WEIGHTS["title_context_overlap"],
        "genre_match": genre_match * _WEIGHTS["genre_match"],
        "domain_match": domain_match * _WEIGHTS["domain_match"],
        "importance": importance * _WEIGHTS["importance"],
        "track_context_relevance": track_context_relevance * _WEIGHTS["track_context_relevance"],
        "workflow_relevance": workflow_relevance * _WEIGHTS["workflow_relevance"],
        "tag_boost": tag_boost,
    }
    final_score = sum(component_scores.values())
    return _CandidateScore(chunk=chunk, final_score=final_score, component_scores=component_scores)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _overlap_ratio(left: set[str], right: set[str], *, cap: int = 4) -> float:
    if not left or not right:
        return 0.0
    return min(len(left & right), cap) / cap


def _metadata_tags(chunk: RetrievedChunk) -> tuple[str, ...]:
    serialized = chunk.metadata.get("tags_serialized", "")
    if not isinstance(serialized, str) or not serialized:
        return ()
    return tuple(part for part in serialized.split("|") if part)


def _title_terms(chunk: RetrievedChunk) -> tuple[str, ...]:
    return tuple(_tokenize(str(chunk.metadata.get("note_title", ""))))


def _context_terms(chunk: RetrievedChunk) -> tuple[str, ...]:
    parts = [
        str(chunk.metadata.get("heading_context", "")),
        str(chunk.metadata.get("arrangement_track_name", "")),
        str(chunk.metadata.get("arrangement_section_name", "")),
        str(chunk.metadata.get("video_title", "")),
        str(chunk.metadata.get("video_section_title", "")),
    ]
    return tuple(term for part in parts for term in _tokenize(part))


def _metadata_terms(chunk: RetrievedChunk) -> tuple[str, ...]:
    parts = [
        str(chunk.metadata.get("import_genre", "")),
        str(chunk.metadata.get("arrangement_genre", "")),
        str(chunk.metadata.get("domain_profile", "")),
        str(chunk.metadata.get("source_type", "")),
        str(chunk.metadata.get("content_category", "")),
    ]
    return tuple(term for part in parts for term in _tokenize(part))


def _track_context_terms(
    track_context: TrackContext | None,
    *,
    section_focus: str,
) -> set[str]:
    if track_context is None:
        return set()
    values: list[str] = []
    if track_context.genre:
        values.append(track_context.genre)
    if track_context.track_name:
        values.append(track_context.track_name)
    values.extend(track_context.vibe)
    values.extend(track_context.reference_tracks)
    if track_context.current_stage:
        values.append(track_context.current_stage)
    if track_context.current_problem:
        values.append(track_context.current_problem)
    values.extend(track_context.known_issues)
    values.extend(track_context.goals)
    normalized_focus = section_focus.strip().lower()
    if normalized_focus and normalized_focus in track_context.sections:
        section = track_context.sections[normalized_focus]
        values.extend(
            [
                section.name,
                section.role,
                section.energy_level,
                section.bars,
                section.notes,
                *section.elements,
                *section.issues,
            ]
        )
    return {term for value in values for term in _tokenize(value)}


def _genre_match_score(
    track_context: TrackContext | None,
    chunk: RetrievedChunk,
    searchable_terms: set[str],
) -> float:
    if track_context is None or not track_context.genre:
        return 0.0
    target_genre = track_context.genre.strip().lower()
    import_genre = str(chunk.metadata.get("import_genre", "")).strip().lower()
    arrangement_genre = str(chunk.metadata.get("arrangement_genre", "")).strip().lower()
    if target_genre and target_genre in {import_genre, arrangement_genre}:
        return 1.0
    genre_terms = set(_tokenize(target_genre))
    if not genre_terms:
        return 0.0
    if genre_terms <= searchable_terms:
        return 0.75
    if genre_terms & searchable_terms:
        return 0.4
    return 0.0


def _domain_match_score(domain_profile: DomainProfile, chunk: RetrievedChunk) -> float:
    metadata_profile = str(chunk.metadata.get("domain_profile", "")).strip().lower()
    if metadata_profile and metadata_profile == domain_profile.value:
        return 1.0
    source_type = str(chunk.metadata.get("source_type", "")).strip().lower()
    if domain_profile == DomainProfile.ELECTRONIC_MUSIC and source_type in _ELECTRONIC_MUSIC_TYPES:
        return 0.4
    return 0.0


def _importance_score(chunk: RetrievedChunk) -> float:
    category = str(chunk.metadata.get("content_category", "")).strip().lower()
    source_kind = str(chunk.metadata.get("source_kind", "")).strip().lower()
    base = _IMPORTANCE_BY_CATEGORY.get(category, 0.0)
    base += _IMPORTANCE_BY_SOURCE_KIND.get(source_kind, 0.0)
    if str(chunk.metadata.get("source_type", "")).strip().lower() == "track_arrangement":
        base += 0.2
    return base


def _track_context_relevance_score(
    track_context_terms: set[str],
    searchable_terms: set[str],
    chunk: RetrievedChunk,
    section_focus: str,
) -> float:
    if not track_context_terms:
        return 0.0
    score = _overlap_ratio(track_context_terms, searchable_terms, cap=5)
    normalized_focus = section_focus.strip().lower()
    if normalized_focus:
        section_terms = {
            term
            for term in _tokenize(
                " ".join(
                    [
                        str(chunk.metadata.get("arrangement_section_name", "")),
                        str(chunk.metadata.get("video_section_title", "")),
                        str(chunk.metadata.get("heading_context", "")),
                    ]
                )
            )
        }
        focus_terms = set(_tokenize(normalized_focus))
        if focus_terms and focus_terms & section_terms:
            score += 0.35
    return min(score, 1.0)


def _workflow_relevance_score(
    collaboration_workflow: CollaborationWorkflow,
    chunk: RetrievedChunk,
) -> float:
    if collaboration_workflow == CollaborationWorkflow.GENERAL_ASK:
        return 0.0
    source_type = str(chunk.metadata.get("source_type", "")).strip().lower()
    return _WORKFLOW_SOURCE_TYPE_BONUSES.get(collaboration_workflow, {}).get(source_type, 0.0)
