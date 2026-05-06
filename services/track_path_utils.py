"""Path-safe filename helpers for per-track persistence files."""

from __future__ import annotations

from hashlib import blake2s
import re

_SAFE_TRACK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_track_file_stem(track_id: str) -> str:
    """Return a traversal-safe, collision-resistant filename stem for a track ID/name."""
    raw_track_id = normalized_track_id(track_id)
    if _SAFE_TRACK_ID_RE.fullmatch(raw_track_id) and raw_track_id not in {".", ".."}:
        return raw_track_id

    path_parts = [
        part
        for part in raw_track_id.replace("\\", "/").split("/")
        if part.strip() and part.strip() not in {".", ".."}
    ]
    flattened = "_".join(path_parts) or "track"
    safe = re.sub(r"[^\w._-]+", "_", flattened, flags=re.UNICODE).strip("._-")
    safe = re.sub(r"_+", "_", safe)
    safe = safe or "track"
    digest = blake2s(raw_track_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"{safe}_{digest}"


def normalized_track_id(track_id: str) -> str:
    """Return the persisted payload value for an entered track ID/name."""
    return (track_id or "default_track").strip() or "default_track"


def legacy_flat_track_file_stem(track_id: str) -> str | None:
    """Return the pre-sanitizer flat filename stem when it cannot traverse directories."""
    raw_track_id = normalized_track_id(track_id)
    if raw_track_id in {".", ".."}:
        return None
    if any(ord(character) < 32 for character in raw_track_id):
        return None
    if "/" in raw_track_id or "\\" in raw_track_id:
        return None
    return raw_track_id
