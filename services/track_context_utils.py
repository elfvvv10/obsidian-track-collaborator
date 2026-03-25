"""Normalization helpers for YAML-backed track context."""

from __future__ import annotations

from collections.abc import Mapping

from services.models import TrackContext


VALID_WORKFLOW_MODES = {
    "general",
    "track_critique",
    "composition",
    "arrangement",
    "sound_design",
    "critique",
    "mixing",
    "research",
}

VALID_CURRENT_STAGES = {
    "idea",
    "sketch",
    "writing",
    "arrangement",
    "sound_design",
    "production",
    "mixing",
    "mastering",
    "finalizing",
}


def _clean_str(value: object) -> str | None:
    """Return a stripped string when present, otherwise None."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_list(value: object) -> list[str]:
    """Normalize a list-like value into a compact string list."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    cleaned: list[str] = []
    for item in items:
        normalized = _clean_str(item)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _clean_dict_str(value: object) -> dict[str, str]:
    """Normalize a mapping into a string-to-string dictionary."""
    if not isinstance(value, Mapping):
        return {}
    cleaned: dict[str, str] = {}
    for key, raw_value in value.items():
        cleaned_key = _clean_str(key)
        cleaned_value = _clean_str(raw_value)
        if cleaned_key and cleaned_value:
            cleaned[cleaned_key] = cleaned_value
    return cleaned


def _coerce_bpm(value: object) -> int | None:
    """Convert BPM values to an integer when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None

    cleaned = _clean_str(value)
    if not cleaned:
        return None
    try:
        bpm = int(float(cleaned))
    except ValueError:
        return None
    return bpm if bpm > 0 else None


def normalize_track_context(raw: dict) -> TrackContext:
    """Normalize raw YAML data into the TrackContext dataclass."""
    track_id = _clean_str(raw.get("track_id")) or "default_track"
    workflow_mode = (_clean_str(raw.get("workflow_mode")) or "general").lower()
    if workflow_mode not in VALID_WORKFLOW_MODES:
        workflow_mode = "general"

    current_stage = _clean_str(raw.get("current_stage"))
    if current_stage is not None:
        current_stage = current_stage.lower()
        if current_stage not in VALID_CURRENT_STAGES:
            current_stage = None

    return TrackContext(
        track_id=track_id,
        track_name=_clean_str(raw.get("track_name")),
        genre=_clean_str(raw.get("genre")),
        bpm=_coerce_bpm(raw.get("bpm")),
        key=_clean_str(raw.get("key")),
        vibe=_clean_list(raw.get("vibe")),
        reference_tracks=_clean_list(raw.get("reference_tracks")),
        workflow_mode=workflow_mode,
        current_stage=current_stage,
        current_section=_clean_str(raw.get("current_section")),
        sections=_clean_dict_str(raw.get("sections")),
        known_issues=_clean_list(raw.get("known_issues")),
        goals=_clean_list(raw.get("goals")),
        notes=_clean_list(raw.get("notes")),
    )
