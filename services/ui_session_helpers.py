"""Pure helpers for Streamlit session UX rendering."""

from __future__ import annotations

from services.models import CollaborationWorkflow, TrackContext, TrackContextSuggestions
from utils import RetrievedChunk


DEV_MODE_PRESET_FAST = "⚡ Fast Dev (Cheap)"
DEV_MODE_PRESET_QUALITY = "🧠 Quality Check"
DEV_MODE_PRESET_LOCAL = "🖥 Local (Ollama)"
DEV_MODE_PRESET_MANUAL = "Manual (Custom)"


def current_track_summary(
    track_context: TrackContext | None,
    *,
    use_track_context: bool = True,
    track_id: str = "",
) -> tuple[str, str, list[tuple[str, str]]]:
    """Return a compact current-track summary suitable for UI rendering."""
    if track_context is None:
        if use_track_context and track_id.strip():
            return (
                "Current Track",
                f"Track memory is enabled for `{track_id.strip()}`, but no YAML Track Context is loaded yet.",
                [],
            )
        if use_track_context:
            return (
                "Current Track",
                "Track memory is enabled, but no Track ID is active yet.",
                [],
            )
        return ("Current Track", "No active YAML Track Context", [])

    display_name = track_context.track_name or track_context.track_id
    rows: list[tuple[str, str]] = [("Track ID", track_context.track_id)]
    optional_fields = (
        ("Track Name", track_context.track_name),
        ("Genre", track_context.genre),
        ("BPM", str(track_context.bpm) if track_context.bpm is not None else ""),
        ("Key", track_context.key),
        ("Vibe", ", ".join(track_context.vibe)),
        ("Reference Tracks", ", ".join(track_context.reference_tracks)),
        ("Current Stage", track_context.current_stage),
        ("Current Problem", track_context.current_problem),
    )
    for label, value in optional_fields:
        if value:
            rows.append((label, value))
    return ("Current Track", f"Using persistent Track Context for `{display_name}`.", rows)


def track_context_status(
    *,
    use_track_context: bool,
    track_id: str,
    existed_before_load: bool,
    track_context: TrackContext | None,
) -> tuple[str, str]:
    """Return compact YAML track-memory status text for the sidebar."""
    if not use_track_context:
        return ("Track memory is off.", "Enable YAML Track Context to load or create persistent track memory.")
    if not track_id.strip():
        return ("Track memory is on, but no Track ID is selected.", "Enter a Track ID to load an existing track or start a new one.")
    if track_context is None:
        return (
            f"Track memory is waiting for `{track_id.strip()}`.",
            "The track will load after a Track ID is provided.",
        )
    if existed_before_load:
        display_name = track_context.track_name or track_context.track_id
        return (
            f"Loaded existing track memory for `{display_name}`.",
            "You are editing a saved YAML Track Context.",
        )
    return (
        f"Started a new track memory for `{track_context.track_id}`.",
        "This is a fresh YAML Track Context. Add details and save when ready.",
    )


def critique_support_summary(
    workflow: CollaborationWorkflow,
    track_context: TrackContext | None,
    chunks: list[RetrievedChunk] | None = None,
) -> tuple[str, list[str]] | None:
    """Describe whether critique is general, track-aware, or arrangement-aware."""
    if workflow != CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE:
        return None

    arrangement_in_play = any(
        str(chunk.metadata.get("source_type", "")).strip().lower() == "track_arrangement"
        for chunk in (chunks or [])
    )
    if track_context is None:
        return (
            "General critique mode",
            [
                "No YAML Track Context is active, so critique will rely on the question and retrieved evidence.",
                "Arrangement-aware support becomes available only when arrangement notes are retrieved.",
            ],
        )
    if arrangement_in_play:
        return (
            "Track-aware critique with arrangement support",
            [
                f"Using persistent Track Context for `{track_context.track_name or track_context.track_id}`.",
                "Arrangement notes were retrieved in the latest answer, so critique can reference sections and bars when useful.",
            ],
        )
    return (
        "Track-aware critique",
        [
            f"Using persistent Track Context for `{track_context.track_name or track_context.track_id}`.",
            "Arrangement-aware support depends on retrieving track arrangement notes for this track.",
        ],
    )


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
    if suggestions.current_stage:
        groups.append(("Current Stage", suggestions.current_stage))
    if suggestions.current_problem:
        groups.append(("Current Problem", suggestions.current_problem))
    return groups


def debug_query_summary(original_question: str, rewritten_query: str) -> list[tuple[str, str]]:
    """Return compact query/debug summary rows."""
    rows = [("Original question", original_question.strip() or "")]
    if rewritten_query.strip():
        rows.append(("Rewritten retrieval query", rewritten_query.strip()))
    return rows


def synced_chat_provider_selection(
    *,
    current_selection: str,
    committed_override: str,
    configured_provider: str,
    last_synced_override: str,
) -> tuple[str, str]:
    """Return the staged provider selection without clobbering in-progress UI edits."""
    committed_selection = (
        committed_override
        if committed_override.strip()
        else f"Use configured default ({configured_provider})"
    )
    if not current_selection.strip() or committed_override.strip() != last_synced_override.strip():
        return committed_selection, committed_override.strip()
    return current_selection, last_synced_override.strip()


def synced_dev_mode_preset_selection(
    *,
    current_selection: str,
    committed_preset: str,
    last_synced_preset: str,
) -> tuple[str, str]:
    """Return the staged preset selection without clobbering in-progress UI edits."""
    committed_selection = committed_preset.strip() or DEV_MODE_PRESET_MANUAL
    if not current_selection.strip() or committed_preset.strip() != last_synced_preset.strip():
        return committed_selection, committed_preset.strip()
    return current_selection, last_synced_preset.strip()


def dev_mode_preset_options() -> list[str]:
    """Return the supported dev-mode preset options in display order."""
    return [
        DEV_MODE_PRESET_MANUAL,
        DEV_MODE_PRESET_FAST,
        DEV_MODE_PRESET_QUALITY,
        DEV_MODE_PRESET_LOCAL,
    ]


def resolve_dev_mode_preset(
    preset: str,
    *,
    configured_ollama_model: str,
    available_ollama_models: list[str] | None = None,
) -> tuple[str, str] | None:
    """Resolve a preset into provider/model overrides for the current session."""
    normalized = preset.strip()
    if not normalized or normalized == DEV_MODE_PRESET_MANUAL:
        return None
    if normalized == DEV_MODE_PRESET_FAST:
        return ("openai", "gpt-4.1-mini")
    if normalized == DEV_MODE_PRESET_QUALITY:
        return ("openai", "gpt-4.1")
    if normalized == DEV_MODE_PRESET_LOCAL:
        default_model = configured_ollama_model.strip()
        if default_model:
            return ("ollama", default_model)
        if available_ollama_models:
            return ("ollama", available_ollama_models[0].strip())
        return ("ollama", "")
    return None
