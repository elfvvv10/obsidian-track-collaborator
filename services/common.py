"""Shared service helpers for indexing and query orchestration."""

from __future__ import annotations

from utils import Note, make_note_key, normalize_path
from vector_store import VectorStore


def ensure_index_compatible(vector_store: VectorStore) -> None:
    """Raise a clear error when the local index format is stale."""
    if vector_store.is_index_compatible():
        return
    raise RuntimeError(
        "The local index format is out of date for this version of the app. "
        "Run `python main.py rebuild` to recreate the index."
    )


def resolve_note_links(notes: list[Note]) -> None:
    """Resolve parsed note links to note keys for later linked-note expansion."""
    alias_map = build_note_alias_map(notes)

    for note in notes:
        own_note_key = make_note_key(note.path)
        resolved_keys: list[str] = []
        seen: set[str] = set()
        for link in note.links:
            note_key = alias_map.get(link)
            if not note_key or note_key == own_note_key or note_key in seen:
                continue
            seen.add(note_key)
            resolved_keys.append(note_key)
        note.linked_note_keys = tuple(resolved_keys)


def build_note_alias_map(notes: list[Note]) -> dict[str, str]:
    """Build a simple alias map for Obsidian note link resolution."""
    alias_map: dict[str, str] = {}

    for note in notes:
        note_key = make_note_key(note.path)
        normalized_path = normalize_path(note.path).lower()
        aliases = {
            normalized_path,
            normalized_path.rsplit(".", 1)[0],
            normalized_path.split("/")[-1],
            normalized_path.split("/")[-1].rsplit(".", 1)[0],
            note.title.strip().lower(),
        }
        for alias in aliases:
            if alias:
                alias_map[alias] = note_key

    return alias_map
