"""Helpers for selecting legacy markdown track contexts from the vault."""

from __future__ import annotations

from pathlib import Path


class TrackSelectorService:
    """Discover track folders that expose a legacy markdown track context."""

    def list_tracks(self, vault_path: Path) -> list[dict[str, str]]:
        """Return sorted relative markdown track-context paths under Projects/."""
        projects_path = vault_path / "Projects"
        if not projects_path.exists() or not projects_path.is_dir():
            return []

        tracks: list[dict[str, str]] = []
        for track_context_path in projects_path.rglob("track_context.md"):
            if not track_context_path.is_file():
                continue
            track_folder = track_context_path.parent
            try:
                relative_folder = track_folder.relative_to(projects_path)
            except ValueError:
                continue
            display_name = " / ".join(relative_folder.parts) or track_folder.name
            tracks.append(
                {
                    "name": display_name,
                    "path": track_context_path.relative_to(vault_path).as_posix(),
                }
            )
        return sorted(tracks, key=lambda item: item["name"].lower())


def selected_track_path(selected_track: str, tracks: list[dict[str, str]]) -> str | None:
    """Resolve a selected track name to its relative markdown path."""
    selected_name = (selected_track or "").strip()
    if selected_name == "None" or not selected_name:
        return None
    for track in tracks:
        if track["name"] == selected_name:
            return track["path"]
    return None


def selected_track_index(current_path: str, tracks: list[dict[str, str]]) -> int:
    """Return the selectbox index for an existing markdown track-context path."""
    normalized_path = (current_path or "").strip()
    if not normalized_path:
        return 0
    for index, track in enumerate(tracks, start=1):
        if track["path"] == normalized_path:
            return index
    return 0
