"""Track-context loading, persistence, and prompt formatting helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from config import AppConfig
from metadata_parser import parse_markdown_metadata
from services.models import CollaborationWorkflow, TrackContext, TrackContextSuggestions
from services.track_context_utils import normalize_track_context
from utils import ensure_directory, get_logger


logger = get_logger()


_TRACK_CONTEXT_FIELDS: tuple[str, ...] = (
    "type",
    "project_id",
    "track_title",
    "primary_genre",
    "secondary_influences",
    "bpm",
    "key",
    "time_signature",
    "vibe",
    "energy_profile",
    "reference_artists",
    "reference_tracks",
    "listener_goal",
    "status",
    "current_section",
    "completion_estimate",
    "last_major_change",
    "current_issues",
    "priority_focus",
    "notes",
    "tags",
)


@dataclass(slots=True)
class TrackContextResult:
    """Parsed track-context payload and prompt-ready block."""

    resolved_path: Path | None = None
    frontmatter: dict[str, object] | None = None
    body: str = ""
    prompt_block: str = ""
    found: bool = False


class TrackContextService:
    """Resolve, parse, format, and persist per-track context documents."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @property
    def yaml_directory(self) -> Path:
        """Return the storage directory for YAML-backed track contexts."""
        return self.config.obsidian_output_path / "track_contexts"

    def exists(self, track_id: str) -> bool:
        """Return whether a YAML track context already exists."""
        return self._yaml_path(track_id).exists()

    def load(self, track_id: str) -> TrackContext:
        """Load and normalize a YAML track context file."""
        path = self._yaml_path(track_id)
        raw_text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not raw_text.strip():
            raw_data: dict[str, object] = {"track_id": track_id}
        else:
            loaded = yaml.safe_load(raw_text)
            raw_data = loaded if isinstance(loaded, dict) else {"track_id": track_id}
            raw_data.setdefault("track_id", track_id)
        context = normalize_track_context(raw_data)
        self._debug_log("Track context YAML load: %s", path)
        return context

    def create_default(self, track_id: str) -> TrackContext:
        """Create a minimal normalized YAML track context on disk."""
        context = normalize_track_context({"track_id": track_id})
        self.save(context)
        return context

    def load_or_create(self, track_id: str) -> TrackContext:
        """Load an existing YAML track context or create a default one."""
        if self.exists(track_id):
            return self.load(track_id)
        return self.create_default(track_id)

    def save(self, context: TrackContext) -> Path:
        """Persist a normalized YAML track context."""
        normalized = normalize_track_context(asdict(context))
        destination = self._yaml_path(normalized.track_id)
        ensure_directory(destination.parent)
        body = yaml.safe_dump(
            asdict(normalized),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        destination.write_text(body, encoding="utf-8")
        self._debug_log("Track context YAML save: %s", destination)
        return destination

    def update_fields(self, track_id: str, updates: dict[str, object]) -> TrackContext:
        """Merge updated fields into an existing YAML track context."""
        existing = asdict(self.load_or_create(track_id))
        existing.update(updates)
        existing["track_id"] = track_id
        context = normalize_track_context(existing)
        self.save(context)
        return context

    def apply_suggestions(
        self,
        track_id: str,
        suggestions: TrackContextSuggestions,
    ) -> TrackContext:
        """Apply reviewed assistant suggestions without overwriting existing list values."""
        context = self.load_or_create(track_id)
        updates: dict[str, object] = {
            "known_issues": _merge_unique(context.known_issues, suggestions.known_issues),
            "goals": _merge_unique(context.goals, suggestions.goals),
        }
        if suggestions.current_stage:
            updates["current_stage"] = suggestions.current_stage
        if suggestions.current_problem:
            updates["current_problem"] = suggestions.current_problem
        return self.update_fields(track_id, updates)

    def get_track_context(
        self,
        workflow: CollaborationWorkflow,
        track_context_path: str | None,
    ) -> TrackContextResult:
        """Return prompt-ready legacy markdown context for supported workflows."""
        if workflow not in {
            CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            CollaborationWorkflow.ARRANGEMENT_PLANNER,
        }:
            return TrackContextResult()

        resolved_path = self._resolve_track_context_path(track_context_path)
        if resolved_path is None:
            self._debug_log(
                "Track context lookup: no track_context_path provided for workflow=%s.",
                workflow.value,
            )
            return TrackContextResult()

        if not resolved_path.exists() or not resolved_path.is_file():
            self._debug_log(
                "Track context lookup: path missing for workflow=%s at %s.",
                workflow.value,
                resolved_path,
            )
            return TrackContextResult(resolved_path=resolved_path)

        raw_content = resolved_path.read_text(encoding="utf-8")
        frontmatter, body = parse_markdown_metadata(raw_content)
        prompt_block = self._format_prompt_block(frontmatter, body)
        self._debug_log(
            "Track context lookup: loaded context for workflow=%s from %s.",
            workflow.value,
            resolved_path,
        )
        return TrackContextResult(
            resolved_path=resolved_path,
            frontmatter=frontmatter,
            body=body,
            prompt_block=prompt_block,
            found=bool(prompt_block),
        )

    def _yaml_path(self, track_id: str) -> Path:
        safe_track_id = (track_id or "default_track").strip() or "default_track"
        return self.yaml_directory / f"{safe_track_id}.yaml"

    def _resolve_track_context_path(self, track_context_path: str | None) -> Path | None:
        raw_path = (track_context_path or "").strip()
        if not raw_path:
            return None

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.config.obsidian_vault_path / candidate).resolve()

        if candidate.suffix.lower() == ".md":
            return candidate
        return (candidate / "track_context.md").resolve()

    def _format_prompt_block(self, frontmatter: dict[str, object], body: str) -> str:
        summary_lines = [
            f"- {field.replace('_', ' ').title()}: {self._format_field_value(frontmatter[field])}"
            for field in _TRACK_CONTEXT_FIELDS
            if field in frontmatter and self._format_field_value(frontmatter[field])
        ]
        body = body.strip()
        body_block = f"\n\nTrack context notes:\n{body}" if body else ""
        if not summary_lines and not body:
            return ""
        summary_block = "\n".join(summary_lines)
        if summary_block:
            summary_block = f"Track context summary:\n{summary_block}"
        instruction = (
            "Use this as internal track-state guidance for continuity, prioritization, and finish-oriented advice. "
            "Do not treat it as evidence or a citation source."
        )
        content_parts = [instruction]
        if summary_block:
            content_parts.append(summary_block)
        if body_block:
            content_parts.append(body_block.lstrip("\n"))
        return "\n\n".join(content_parts)

    def _format_field_value(self, value: object) -> str:
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    def _debug_log(self, message: str, *args: object) -> None:
        if self.config.framework_debug:
            logger.info(message, *args)


def _merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = {item.strip().lower() for item in existing if item.strip()}
    for item in additions:
        cleaned = item.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        merged.append(cleaned)
    return merged
