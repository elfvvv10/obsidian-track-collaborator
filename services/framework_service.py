"""Framework document loading for workflow-aware prompt injection."""

from __future__ import annotations

from pathlib import Path

from config import AppConfig
from services.models import CollaborationWorkflow, DomainProfile
from utils import get_logger


logger = get_logger()


class FrameworkService:
    """Resolve and load framework documents for specific workflows."""

    _FRAMEWORK_REGISTRY: dict[CollaborationWorkflow, str] = {
        CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE: "track_critique_framework_v1.md",
    }

    def __init__(self, config: AppConfig, *, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parents[1]
        self._framework_cache: dict[Path, str] = {}

    def get_framework_text(
        self,
        workflow: CollaborationWorkflow,
        domain_profile: DomainProfile,
    ) -> str:
        """Return framework text for a workflow, or an empty string when none is available."""
        del domain_profile  # Reserved for future domain-specific framework routing.
        framework_path, resolution_source = self._resolve_framework_path(workflow)
        if framework_path is None:
            self._debug_log(
                "Framework lookup: no registered framework for workflow=%s.",
                workflow.value,
            )
            return ""

        if not framework_path.exists() or not framework_path.is_file():
            self._debug_log(
                "Framework lookup: %s path missing for workflow=%s at %s.",
                resolution_source,
                workflow.value,
                framework_path,
            )
            return ""

        cached_text = self._framework_cache.get(framework_path)
        if cached_text is not None:
            self._debug_log(
                "Framework lookup: cache hit for workflow=%s using %s path %s.",
                workflow.value,
                resolution_source,
                framework_path,
            )
            return cached_text

        self._debug_log(
            "Framework lookup: reading %s path for workflow=%s at %s.",
            resolution_source,
            workflow.value,
            framework_path,
        )
        framework_text = framework_path.read_text(encoding="utf-8").strip()
        self._framework_cache[framework_path] = framework_text
        return framework_text

    def _resolve_framework_path(self, workflow: CollaborationWorkflow) -> tuple[Path | None, str]:
        filename = self._FRAMEWORK_REGISTRY.get(workflow)
        if not filename:
            return None, "unregistered"

        override_path = self._resolve_override_path(workflow)
        if override_path is not None and override_path.exists():
            return override_path, "override"

        candidates: tuple[tuple[Path, str], ...] = (
            (self.config.obsidian_vault_path / "Sources" / "Frameworks", "vault-sources"),
            (self.config.obsidian_vault_path / "Templates" / "Frameworks", "vault-legacy-templates"),
            (self.config.obsidian_vault_path / "Knowledge" / "Frameworks", "vault-legacy-knowledge"),
            (self.repo_root / "sources" / "frameworks", "repo-sources"),
            (self.repo_root / "templates" / "frameworks", "repo-legacy-default"),
            (self.repo_root / "knowledge" / "frameworks", "repo-legacy-default"),
        )
        for directory, source in candidates:
            discovered = self._resolve_framework_from_directory(directory, filename)
            if discovered is not None:
                return discovered, source

        return (self.repo_root / "templates" / "frameworks" / filename).resolve(), "repo-legacy-default"

    def _resolve_framework_from_directory(self, directory: Path, filename: str) -> Path | None:
        if not directory.exists() or not directory.is_dir():
            return None
        direct_path = (directory / filename).resolve()
        if direct_path.exists():
            return direct_path
        matches = sorted(directory.rglob(filename))
        if matches:
            return matches[0].resolve()
        return None

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

    def _debug_log(self, message: str, *args: object) -> None:
        if self.config.framework_debug:
            logger.info(message, *args)
