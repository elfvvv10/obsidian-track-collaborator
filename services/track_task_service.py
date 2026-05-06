"""Persisted per-track task helpers tied to canonical YAML Track Context."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import yaml

from config import AppConfig
from services.models import PersistedTrackTask, SessionTask
from services.track_path_utils import legacy_flat_track_file_stem, safe_track_file_stem
from utils import ensure_directory, get_logger, current_timestamp


logger = get_logger()

_VALID_STATUSES = {"open", "done", "deferred"}
_VALID_PRIORITIES = {"low", "medium", "high"}


class TrackTaskService:
    """Load, save, and mutate simple per-track task files stored beside Track Context."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @property
    def task_directory(self) -> Path:
        return self.config.obsidian_output_path / "track_contexts"

    def task_path(self, track_id: str) -> Path:
        safe_track_id = safe_track_file_stem(track_id)
        return self.task_directory / f"{safe_track_id}.tasks.yaml"

    def existing_task_path(self, track_id: str) -> Path:
        safe_path = self.task_path(track_id)
        if safe_path.exists():
            return safe_path
        legacy_stem = legacy_flat_track_file_stem(track_id)
        if legacy_stem is None:
            return safe_path
        legacy_path = self.task_directory / f"{legacy_stem}.tasks.yaml"
        return legacy_path if legacy_path.exists() else safe_path

    def load_tasks(self, track_id: str) -> list[PersistedTrackTask]:
        """Load persisted tasks for a track, returning normalized defaults when absent."""
        path = self.existing_task_path(track_id)
        raw_text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not raw_text.strip():
            return []
        loaded = yaml.safe_load(raw_text)
        if not isinstance(loaded, dict):
            return []
        raw_tasks = loaded.get("tasks", [])
        if not isinstance(raw_tasks, list):
            return []
        return [_normalize_task(item) for item in raw_tasks if isinstance(item, dict)]

    def save_tasks(self, track_id: str, tasks: list[PersistedTrackTask]) -> Path:
        """Persist the full task list for a track."""
        path = self.task_path(track_id)
        ensure_directory(path.parent)
        payload = {
            "track_id": (track_id or "default_track").strip() or "default_track",
            "schema_version": "track_tasks_v1",
            "tasks": [self._serialize_task(task) for task in tasks],
        }
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        self._debug_log("Track task YAML save: %s", path)
        return path

    def add_task(
        self,
        track_id: str,
        *,
        text: str,
        created_from: str = "user",
        priority: str = "medium",
        linked_section: str = "",
        notes: str = "",
    ) -> PersistedTrackTask:
        """Create and persist a new task for the given track."""
        tasks = self.load_tasks(track_id)
        task = PersistedTrackTask(
            id=uuid4().hex,
            text=text.strip(),
            status="open",
            priority=_normalize_priority(priority),
            linked_section=(linked_section or "").strip(),
            created_from=(created_from or "user").strip() or "user",
            created_at=current_timestamp(),
            notes=(notes or "").strip(),
        )
        tasks.append(task)
        self.save_tasks(track_id, tasks)
        return task

    def update_task(self, track_id: str, task_id: str, updates: dict[str, object]) -> PersistedTrackTask | None:
        """Update a persisted task in place and save the containing file."""
        tasks = self.load_tasks(track_id)
        updated_task: PersistedTrackTask | None = None
        updated_tasks: list[PersistedTrackTask] = []
        for task in tasks:
            if task.id != task_id:
                updated_tasks.append(task)
                continue
            raw = asdict(task)
            raw.update(updates)
            if "priority" in raw:
                raw["priority"] = _normalize_priority(raw["priority"])
            if "status" in raw:
                raw["status"] = _normalize_status(raw["status"])
            if raw["status"] != "done":
                raw["completed_at"] = None
            raw["text"] = str(raw.get("text", "")).strip()
            raw["linked_section"] = str(raw.get("linked_section", "") or "").strip()
            raw["created_from"] = str(raw.get("created_from", "user") or "user").strip() or "user"
            raw["notes"] = str(raw.get("notes", "") or "").strip()
            updated_task = _normalize_task(raw)
            updated_tasks.append(updated_task)
        if updated_task is None:
            return None
        self.save_tasks(track_id, updated_tasks)
        return updated_task

    def complete_task(
        self,
        track_id: str,
        task_id: str,
        *,
        completed: bool = True,
    ) -> PersistedTrackTask | None:
        """Mark a task completed or reopened and persist the updated timestamp."""
        return self.update_task(
            track_id,
            task_id,
            {
                "status": "done" if completed else "open",
                "completed_at": current_timestamp() if completed else None,
            },
        )

    def delete_task(self, track_id: str, task_id: str) -> bool:
        """Delete a task from the persisted track task list."""
        tasks = self.load_tasks(track_id)
        kept = [task for task in tasks if task.id != task_id]
        if len(kept) == len(tasks):
            return False
        self.save_tasks(track_id, kept)
        return True

    def load_session_tasks(self, track_id: str) -> list[SessionTask]:
        """Return prompt-friendly task rows derived from persisted track tasks."""
        return [self.to_session_task(task) for task in self.load_tasks(track_id)]

    def to_session_task(self, task: PersistedTrackTask) -> SessionTask:
        """Adapt a persisted task into the existing prompt-facing task shape."""
        return SessionTask(
            id=task.id,
            text=task.text,
            status=task.status,
            source=task.created_from,
            created_at=task.created_at,
            notes=task.notes,
            priority=task.priority,
            linked_section=task.linked_section,
            completed_at=task.completed_at,
        )

    def _serialize_task(self, task: PersistedTrackTask) -> dict[str, object]:
        return {
            "id": task.id,
            "text": task.text,
            "status": task.status,
            "priority": task.priority,
            "linked_section": task.linked_section,
            "created_from": task.created_from,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
            "notes": task.notes,
        }

    def _debug_log(self, message: str, *args: object) -> None:
        if self.config.framework_debug:
            logger.info(message, *args)


def _normalize_task(raw: dict[str, object]) -> PersistedTrackTask:
    status = _normalize_status(raw.get("status"))
    completed_at = str(raw.get("completed_at", "")).strip() or None
    if status != "done":
        completed_at = None
    return PersistedTrackTask(
        id=str(raw.get("id", "")).strip() or uuid4().hex,
        text=str(raw.get("text", "")).strip(),
        status=status,
        priority=_normalize_priority(raw.get("priority")),
        linked_section=str(raw.get("linked_section", "") or "").strip(),
        created_from=str(raw.get("created_from", "user") or "user").strip() or "user",
        created_at=str(raw.get("created_at", "")).strip() or current_timestamp(),
        completed_at=completed_at,
        notes=str(raw.get("notes", "") or "").strip(),
    )


def _normalize_status(value: object) -> str:
    normalized = str(value or "open").strip().lower()
    if normalized == "completed":
        return "done"
    return normalized if normalized in _VALID_STATUSES else "open"


def _normalize_priority(value: object) -> str:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in _VALID_PRIORITIES else "medium"
