"""Canonical YAML Track Context helpers with legacy markdown compatibility bridges."""

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
class LegacyTrackContextResult:
    """Parsed legacy markdown track-context payload and prompt-ready block."""

    resolved_path: Path | None = None
    frontmatter: dict[str, object] | None = None
    body: str = ""
    prompt_block: str = ""
    found: bool = False


TrackContextResult = LegacyTrackContextResult


class TrackContextService:
    """Load/save canonical YAML Track Context and bridge legacy markdown imports."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @property
    def yaml_directory(self) -> Path:
        """Return the storage directory for canonical YAML Track Context files."""
        return self.config.obsidian_output_path / "track_contexts"

    def exists(self, track_id: str) -> bool:
        """Compatibility wrapper for checking canonical YAML Track Context existence."""
        return self.canonical_exists(track_id)

    def canonical_exists(self, track_id: str) -> bool:
        """Return whether a canonical YAML Track Context already exists."""
        return self._yaml_path(track_id).exists()

    def load(self, track_id: str) -> TrackContext:
        """Compatibility wrapper for canonical YAML Track Context loading."""
        return self.load_canonical_track_context(track_id)

    def load_canonical_track_context(self, track_id: str) -> TrackContext:
        """Load and normalize a canonical YAML Track Context file."""
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
        """Compatibility wrapper for canonical YAML Track Context creation."""
        return self.create_default_canonical_track_context(track_id)

    def create_default_canonical_track_context(self, track_id: str) -> TrackContext:
        """Create a minimal canonical YAML Track Context on disk."""
        context = normalize_track_context({"track_id": track_id})
        self.save_canonical_track_context(context)
        return context

    def load_or_create(self, track_id: str) -> TrackContext:
        """Compatibility wrapper for canonical YAML Track Context load/create."""
        return self.load_or_create_canonical_track_context(track_id)

    def load_or_create_canonical_track_context(self, track_id: str) -> TrackContext:
        """Load an existing canonical YAML Track Context or create a default one."""
        if self.canonical_exists(track_id):
            return self.load_canonical_track_context(track_id)
        return self.create_default_canonical_track_context(track_id)

    def save(self, context: TrackContext) -> Path:
        """Compatibility wrapper for canonical YAML Track Context persistence."""
        return self.save_canonical_track_context(context)

    def save_canonical_track_context(self, context: TrackContext) -> Path:
        """Persist a normalized canonical YAML Track Context."""
        normalized = normalize_track_context(asdict(context))
        destination = self._yaml_path(normalized.track_id)
        ensure_directory(destination.parent)
        body = yaml.safe_dump(
            self._serialize_yaml_context(normalized),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        destination.write_text(body, encoding="utf-8")
        self._debug_log("Track context YAML save: %s", destination)
        return destination

    def update_fields(self, track_id: str, updates: dict[str, object]) -> TrackContext:
        """Compatibility wrapper for canonical YAML Track Context updates."""
        return self.update_canonical_track_context_fields(track_id, updates)

    def update_canonical_track_context_fields(
        self,
        track_id: str,
        updates: dict[str, object],
    ) -> TrackContext:
        """Merge updated fields into an existing canonical YAML Track Context."""
        existing = asdict(self.load_or_create_canonical_track_context(track_id))
        existing.update(updates)
        existing["track_id"] = track_id
        context = normalize_track_context(existing)
        self.save_canonical_track_context(context)
        return context

    def apply_suggestions(
        self,
        track_id: str,
        suggestions: TrackContextSuggestions,
    ) -> TrackContext:
        """Apply reviewed assistant suggestions to canonical YAML Track Context."""
        context = self.load_or_create_canonical_track_context(track_id)
        updates: dict[str, object] = {
            "known_issues": _merge_unique(context.known_issues, suggestions.known_issues),
            "goals": _merge_unique(context.goals, suggestions.goals),
            "vibe": _merge_unique(context.vibe, suggestions.vibe_suggestions),
            "reference_tracks": _merge_unique(context.reference_tracks, suggestions.reference_track_suggestions),
        }
        if suggestions.current_stage:
            updates["current_stage"] = suggestions.current_stage
        if suggestions.current_problem:
            updates["current_problem"] = suggestions.current_problem
        if suggestions.bpm_suggestion is not None:
            updates["bpm"] = suggestions.bpm_suggestion
        if suggestions.key_suggestion:
            updates["key"] = suggestions.key_suggestion
        return self.update_canonical_track_context_fields(track_id, updates)

    def get_track_context(
        self,
        workflow: CollaborationWorkflow,
        track_context_path: str | None,
    ) -> LegacyTrackContextResult:
        """Compatibility wrapper for loading legacy markdown workflow context."""
        return self.load_legacy_markdown_context(workflow, track_context_path)

    def load_legacy_markdown_context(
        self,
        workflow: CollaborationWorkflow,
        track_context_path: str | None,
    ) -> LegacyTrackContextResult:
        """Return prompt-ready legacy markdown context for compatibility workflows only."""
        if workflow not in {
            CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            CollaborationWorkflow.ARRANGEMENT_PLANNER,
        }:
            return LegacyTrackContextResult()

        resolved_path = self.resolve_legacy_track_context_path(track_context_path)
        if resolved_path is None:
            self._debug_log(
                "Legacy track context lookup: no track_context_path provided for workflow=%s.",
                workflow.value,
            )
            return LegacyTrackContextResult()

        if not resolved_path.exists() or not resolved_path.is_file():
            self._debug_log(
                "Legacy track context lookup: path missing for workflow=%s at %s.",
                workflow.value,
                resolved_path,
            )
            return LegacyTrackContextResult(resolved_path=resolved_path)

        frontmatter, body = self.parse_legacy_markdown_track_context(resolved_path)
        prompt_block = self._format_prompt_block(frontmatter, body)
        self._debug_log(
            "Legacy track context lookup: loaded compatibility context for workflow=%s from %s.",
            workflow.value,
            resolved_path,
        )
        return LegacyTrackContextResult(
            resolved_path=resolved_path,
            frontmatter=frontmatter,
            body=body,
            prompt_block=prompt_block,
            found=bool(prompt_block),
        )

    def parse_legacy_markdown_track_context(
        self,
        path_or_track_context_path: Path | str,
    ) -> tuple[dict[str, object], str]:
        """Parse a legacy markdown Track Context file into frontmatter and body."""
        resolved_path = (
            path_or_track_context_path
            if isinstance(path_or_track_context_path, Path)
            else self.resolve_legacy_track_context_path(path_or_track_context_path)
        )
        if resolved_path is None:
            return {}, ""
        raw_content = resolved_path.read_text(encoding="utf-8")
        return parse_markdown_metadata(raw_content)

    def import_legacy_markdown_track_context(
        self,
        track_id: str,
        track_context_path: str,
    ) -> TrackContext:
        """Normalize a legacy markdown Track Context into the canonical YAML shape."""
        frontmatter, _body = self.parse_legacy_markdown_track_context(track_context_path)
        normalized = normalize_track_context(
            {
                "track_id": track_id,
                "track_name": frontmatter.get("track_title"),
                "genre": frontmatter.get("primary_genre"),
                "bpm": frontmatter.get("bpm"),
                "key": frontmatter.get("key"),
                "vibe": _coerce_string_list(frontmatter.get("vibe")),
                "reference_tracks": _coerce_string_list(frontmatter.get("reference_tracks")),
                "current_stage": _normalize_legacy_stage(frontmatter.get("status")),
                "current_problem": _first_item(frontmatter.get("current_issues")),
                "known_issues": _coerce_string_list(frontmatter.get("current_issues")),
                "goals": _coerce_string_list(frontmatter.get("priority_focus")),
            }
        )
        return normalized

    def migrate_legacy_markdown_to_canonical_yaml(
        self,
        track_id: str,
        track_context_path: str,
        *,
        overwrite: bool = False,
    ) -> TrackContext:
        """Persist a legacy markdown Track Context into canonical YAML form."""
        if self.canonical_exists(track_id) and not overwrite:
            return self.load_canonical_track_context(track_id)
        migrated = self.import_legacy_markdown_track_context(track_id, track_context_path)
        self.save_canonical_track_context(migrated)
        return migrated

    def _yaml_path(self, track_id: str) -> Path:
        safe_track_id = (track_id or "default_track").strip() or "default_track"
        return self.yaml_directory / f"{safe_track_id}.yaml"

    def _resolve_track_context_path(self, track_context_path: str | None) -> Path | None:
        return self.resolve_legacy_track_context_path(track_context_path)

    def resolve_legacy_track_context_path(self, track_context_path: str | None) -> Path | None:
        """Resolve a legacy markdown Track Context path from the vault."""
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
            "Use this legacy markdown context as compatibility-only internal track-state guidance. "
            "Canonical track state lives in the YAML Track Context system. "
            "Do not treat this as evidence or a citation source."
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

    def _serialize_yaml_context(self, context: TrackContext) -> dict[str, object]:
        """Return the canonical persisted YAML shape for v1 Track Context files."""
        return {
            "track_id": context.track_id,
            "title": context.track_name,
            "genre": context.genre,
            "bpm": context.bpm,
            "key": context.key,
            "vibe": context.vibe,
            "references": context.reference_tracks,
            "current_stage": context.current_stage,
            "current_problem": context.current_problem,
            "known_issues": context.known_issues,
            "goals": context.goals,
            "sections": {
                section_key: {
                    "name": section.name,
                    "bars": section.bars,
                    "role": section.role,
                    "energy_level": section.energy_level,
                    "elements": section.elements,
                    "issues": section.issues,
                    "notes": section.notes,
                }
                for section_key, section in context.sections.items()
            },
        }

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


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    cleaned = str(value).strip() if value is not None else ""
    return [cleaned] if cleaned else []


def _first_item(value: object) -> str | None:
    items = _coerce_string_list(value)
    return items[0] if items else None


def _normalize_legacy_stage(value: object) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return None
    if "arrang" in text:
        return "arrangement"
    if "writ" in text or "draft" in text or "idea" in text:
        return "writing"
    if "mix" in text:
        return "mixing"
    if "sound" in text:
        return "sound_design"
    if "finish" in text or "complete" in text:
        return "finalizing"
    return None
