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
            vibe_suggestions=self._extract_items(answer, _VIBE_PATTERNS, existing=track_context.vibe, limit=3),
            reference_track_suggestions=self._extract_items(
                answer, _REFERENCE_PATTERNS, existing=track_context.reference_tracks, limit=2
            ),
            section_suggestions=self._extract_items(answer, _SECTION_PATTERNS, existing=[], limit=3),
            section_focus=self._extract_section_focus(answer),
            bpm_suggestion=self._extract_bpm(answer),
            key_suggestion=self._extract_key(answer),
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

    def _extract_section_focus(self, answer: str) -> str | None:
        lowered = answer.lower()
        for focus in _SECTION_FOCUS_NAMES:
            escaped = re.escape(focus)
            section_patterns = (
                rf"\bfocus\s+(?:on|is)\s+(?:the\s+)?{escaped}\b",
                rf"\b(?:on|in|for)\s+(?:the\s+)?{escaped}\b",
                rf"\b(?:the|this|that)\s+{escaped}\b",
                rf"(?:^|\n)\s*{escaped}\s+(?:needs|section|is|feels)\b",
            )
            if any(re.search(pattern, lowered) for pattern in section_patterns):
                return focus
        return None

    def _extract_bpm(self, answer: str) -> int | None:
        lowered = answer.lower()
        for pattern in _BPM_PATTERNS:
            match = pattern.search(lowered)
            if match:
                try:
                    value = int(match.group(1))
                    if 20 <= value <= 300:
                        return value
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_key(self, answer: str) -> str | None:
        match = _KEY_RE.search(answer)
        if match:
            raw = match.group(1).strip()
            normalized = raw[0].upper() + raw[1:] if raw else raw
            if normalized:
                return normalized
        return None


_ISSUE_PATTERNS = (
    re.compile(r"^(?:issue|problem|weakness)\s*[:\\-]\s*(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?!\s*(?:issue|problem|weakness)\s*:)(.+\b(?:lacks contrast|feels flat|needs more movement|is muddy|is weak))$",
        re.IGNORECASE,
    ),
)

_GOAL_PATTERNS = (
    re.compile(r"^(?:goal|aim|priority|focus)\s*[:\\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^((?:increase|improve|add|create)\s+.+)$", re.IGNORECASE),
)

_PROBLEM_PATTERNS = (
    re.compile(r"^(?:problem|focus|current problem)\s*[:\\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^(?:note|remember|consider|try)\s*[:\\-]\s*(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?!\s*(?:issue|problem|weakness|goal|aim|priority|focus|current problem|note|remember|consider|try)\s*:)"
        r"(.+\bmay need\b.+)$",
        re.IGNORECASE,
    ),
)

_VIBE_PATTERNS = (
    re.compile(r"^vibe\s*[:\\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^mood\s*[:\\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^feeling\s*[:\\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(
        r"^(dark|driving|energetic|chilled|melodic|atmospheric|aggressive|deep|warm|bright|"
        r"glitchy|industrial|ethereal|groovy|hypnotic|minimal|ambient|punchy|lush|"
        r"resonant|textured|layered|sparse|dense|organic|cinematic|dreamy|"
        r"euphoric|uplifting|tense|brooding|playful|funky|soulful|raw)"
        r"(?:\s+(?:and|&)\s+\w+)?(?:\s+.*)?$",
        re.IGNORECASE,
    ),
)

_REFERENCE_PATTERNS = (
    re.compile(r"^(?:reference track|reference|similar to|reminds me of|like)\s*[:\\-]?\s*(.+)$", re.IGNORECASE),
)

_SECTION_PATTERNS = (
    re.compile(r"^section\s*[:\\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^part\s*[:\\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^focused on\s*(.+)$", re.IGNORECASE),
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

_SECTION_FOCUS_NAMES = (
    "main groove",
    "intro",
    "buildup",
    "build",
    "drop",
    "breakdown",
    "break",
    "verse",
    "chorus",
    "bridge",
    "outro",
    "groove",
    "bassline",
    "transition",
)

_BPM_PATTERNS = (
    re.compile(r"\b(\d{2,3})\s*(?:bpm|beats per minute)\b", re.IGNORECASE),
    re.compile(r"\bbpm\s*[:\\-]?\s*(\d{2,3})\b", re.IGNORECASE),
    re.compile(r"\btempo\s*[:\\-]?\s*(\d{2,3})\b", re.IGNORECASE),
)

_KEY_RE = re.compile(
    r"(?:key\s+of\s+|[\[\(])([A-Ga-g][#b]?(?:m|min|major|minor)?)",
    re.IGNORECASE,
)
