"""Helpers for Knowledge-folder category discovery and normalization."""

from __future__ import annotations

from config import AppConfig


GENERIC_KNOWLEDGE_CATEGORY_LABEL = "Generic advice (no Knowledge category)"


class KnowledgeCategoryService:
    """Resolve Knowledge categories for ingestion UI and CLI validation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def available_categories(self) -> list[str]:
        base_path = self.config.curated_knowledge_path
        if not base_path.exists() or not base_path.is_dir():
            return []
        return [
            path.name
            for path in sorted(base_path.iterdir(), key=lambda item: item.name.casefold())
            if path.is_dir() and not path.name.startswith(".")
        ]

    def display_options(self) -> list[str]:
        return [GENERIC_KNOWLEDGE_CATEGORY_LABEL, *self.available_categories()]

    def canonicalize(self, value: str | None) -> str | None:
        normalized = " ".join((value or "").split()).strip()
        if not normalized or normalized == GENERIC_KNOWLEDGE_CATEGORY_LABEL:
            return None
        for candidate in self.available_categories():
            if candidate.casefold() == normalized.casefold():
                return candidate
        return normalized

    def validate_or_raise(self, value: str | None) -> str | None:
        canonical = self.canonicalize(value)
        if canonical is None:
            return None
        if any(candidate.casefold() == canonical.casefold() for candidate in self.available_categories()):
            return canonical
        available = ", ".join(self.available_categories()) or "none found"
        raise ValueError(
            f"Unknown Knowledge category '{canonical}'. Available Knowledge categories: {available}."
        )
