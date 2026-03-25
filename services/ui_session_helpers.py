"""Pure helpers for Streamlit session UX rendering."""

from __future__ import annotations

from services.models import TrackContext, TrackContextSuggestions


def current_track_summary(track_context: TrackContext | None) -> tuple[str, list[tuple[str, str]]]:
    """Return a compact current-track summary suitable for UI rendering."""
    if track_context is None:
        return "No active YAML Track Context", []

    rows: list[tuple[str, str]] = [("Track ID", track_context.track_id)]
    optional_fields = (
        ("Track Name", track_context.track_name),
        ("Genre", track_context.genre),
        ("BPM", str(track_context.bpm) if track_context.bpm is not None else ""),
        ("Key", track_context.key),
        ("Workflow Mode", track_context.workflow_mode),
        ("Current Stage", track_context.current_stage),
        ("Current Section", track_context.current_section),
    )
    for label, value in optional_fields:
        if value:
            rows.append((label, value))
    return "Active YAML Track Context", rows


def suggestion_groups(
    suggestions: TrackContextSuggestions | None,
) -> list[tuple[str, list[str] | str]]:
    """Return grouped suggestion content for compact display."""
    if suggestions is None:
        return []

    groups: list[tuple[str, list[str] | str]] = []
    if suggestions.known_issues:
        groups.append(("Known Issues", suggestions.known_issues))
    if suggestions.goals:
        groups.append(("Goals", suggestions.goals))
    if suggestions.notes:
        groups.append(("Notes", suggestions.notes))
    if suggestions.current_stage:
        groups.append(("Current Stage", suggestions.current_stage))
    if suggestions.current_section:
        groups.append(("Current Section", suggestions.current_section))
    return groups


def debug_query_summary(original_question: str, rewritten_query: str) -> list[tuple[str, str]]:
    """Return compact query/debug summary rows."""
    rows = [("Original question", original_question.strip() or "")]
    if rewritten_query.strip():
        rows.append(("Rewritten retrieval query", rewritten_query.strip()))
    return rows
