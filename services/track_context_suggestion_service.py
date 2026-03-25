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
            current_stage=self._extract_stage(answer),
            current_problem=self._extract_problem(answer, existing=track_context.current_problem),
        )

        if suggestions.current_stage and suggestions.current_stage == track_context.current_stage:
            suggestions.current_stage = None
        if suggestions.current_problem and suggestions.current_problem == track_context.current_problem:
            suggestions.current_problem = None

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

    def _extract_problem(self, answer: str, *, existing: str | None) -> str | None:
        candidates = self._extract_items(answer, _PROBLEM_PATTERNS, existing=[existing or ""], limit=1)
        return candidates[0] if candidates else None


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

_PROBLEM_PATTERNS = (
    re.compile(r"^(?:problem|focus|current problem)\s*[:\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^(?:note|remember|consider|try)\s*[:\-]\s*(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?!\s*(?:issue|problem|weakness|goal|aim|priority|focus|current problem|note|remember|consider|try)\s*:)"
        r"(.+\bmay need\b.+)$",
        re.IGNORECASE,
    ),
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
