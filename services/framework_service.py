"""Framework document loading for workflow-aware prompt injection."""

from __future__ import annotations

from pathlib import Path

from config import AppConfig
from services.models import CollaborationWorkflow, DomainProfile


class FrameworkService:
    """Resolve and load framework documents for specific workflows."""

    _FRAMEWORK_REGISTRY: dict[CollaborationWorkflow, str] = {
        CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE: "track_critique_framework_v1.md",
    }

    def __init__(self, config: AppConfig, *, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parents[1]

    def get_framework_text(
        self,
        workflow: CollaborationWorkflow,
        domain_profile: DomainProfile,
    ) -> str:
        """Return framework text for a workflow, or an empty string when none is available."""
        del domain_profile  # Reserved for future domain-specific framework routing.
        framework_path = self._resolve_framework_path(workflow)
        if framework_path is None or not framework_path.exists() or not framework_path.is_file():
            return ""
        return framework_path.read_text(encoding="utf-8").strip()

    def _resolve_framework_path(self, workflow: CollaborationWorkflow) -> Path | None:
        filename = self._FRAMEWORK_REGISTRY.get(workflow)
        if not filename:
            return None

        override_path = self._resolve_override_path(workflow)
        if override_path is not None and override_path.exists():
            return override_path
        return self.repo_root / "knowledge" / "frameworks" / filename

    def _resolve_override_path(self, workflow: CollaborationWorkflow) -> Path | None:
        if workflow != CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE:
            return None
        raw_path = self.config.track_critique_framework_path.strip()
        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.repo_root / candidate).resolve()
