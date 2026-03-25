"""Heuristic assistant-suggested Track Context updates."""

from __future__ import annotations

import re

from services.models import TrackContext, TrackContextSuggestions


class TrackContextSuggestionService:
    """Generate conservative, reviewable Track Context suggestions from the answer."""

    def suggest(
        self,
        answer: str,
        track_context: TrackContext | None,
    ) -> TrackContextSuggestions | None:
        """Return compact suggestions or None when nothing useful is found."""
        if track_context is None:
            return None

        suggestions = TrackContextSuggestions(
            known_issues=self._extract_items(answer, _ISSUE_PATTERNS, existing=track_context.known_issues, limit=2),
            goals=self._extract_items(answer, _GOAL_PATTERNS, existing=track_context.goals, limit=2),
            notes=self._extract_items(answer, _NOTE_PATTERNS, existing=track_context.notes, limit=2),
            current_stage=self._extract_stage(answer),
            current_section=self._extract_section(answer),
        )

        if suggestions.current_stage and suggestions.current_stage == track_context.current_stage:
            suggestions.current_stage = None
        if suggestions.current_section and suggestions.current_section == track_context.current_section:
            suggestions.current_section = None

        return None if suggestions.is_empty() else suggestions

    def _extract_items(
        self,
        answer: str,
        patterns: tuple[re.Pattern[str], ...],
        *,
        existing: list[str],
        limit: int,
    ) -> list[str]:
        found: list[str] = []
        existing_normalized = {item.strip().lower() for item in existing if item.strip()}
        for line in answer.splitlines():
            normalized_line = line.strip().lstrip("-* ").strip()
            if not normalized_line or len(normalized_line) < 8:
                continue
            for pattern in patterns:
                match = pattern.search(normalized_line)
                if not match:
                    continue
                candidate = match.group(1).strip(" .:")
                if not candidate:
                    continue
                lowered = candidate.lower()
                if lowered in existing_normalized or lowered in {item.lower() for item in found}:
                    continue
                found.append(candidate)
                if len(found) >= limit:
                    return found
        return found

    def _extract_stage(self, answer: str) -> str | None:
        lowered = answer.lower()
        for stage in _STAGES:
            if re.search(rf"\b{re.escape(stage)}\b", lowered):
                return stage
        return None

    def _extract_section(self, answer: str) -> str | None:
        lowered = answer.lower()
        for section in _SECTIONS:
            if re.search(rf"\b(?:in|during|before|after|around)\s+the\s+{re.escape(section)}\b", lowered):
                return section
            if re.search(rf"\b{re.escape(section)}\s+section\b", lowered):
                return section
        return None


_ISSUE_PATTERNS = (
    re.compile(r"^(?:issue|problem|weakness)\s*[:\-]\s*(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?!\s*(?:issue|problem|weakness)\s*:)(.+\b(?:lacks contrast|feels flat|needs more movement|is muddy|is weak))$",
        re.IGNORECASE,
    ),
)

_GOAL_PATTERNS = (
    re.compile(r"^(?:goal|aim|priority|focus)\s*[:\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^((?:increase|improve|add|create)\s+.+)$", re.IGNORECASE),
)

_NOTE_PATTERNS = (
    re.compile(r"^(?:note|remember|consider|try)\s*[:\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^(?!\s*(?:note|remember|consider|try)\s*:)(.+\bmay need\b.+)$", re.IGNORECASE),
)

_STAGES = (
    "idea",
    "sketch",
    "writing",
    "arrangement",
    "sound_design",
    "production",
    "mixing",
    "mastering",
    "finalizing",
)

_SECTIONS = (
    "intro",
    "breakdown",
    "drop",
    "first drop",
    "second drop",
    "outro",
)
