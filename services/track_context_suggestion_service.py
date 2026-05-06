"""Enhanced assistant-suggested Track Context updates."""
from __future__ import annotations

import re

from services.models import TrackContext, TrackContextSuggestions


# ── Known section names from Track Context and common usage ──
_SECTION_NAMES = {
    "main groove", "intro", "buildup", "build", "drop", "breakdown", "break", "groove",
    "verse", "chorus", "bridge", "outro", "climax", "transition",
    "pre-drop", "post-drop", "mid-section", "interlude", "riff",
    "main", "hook", "pad", "bass", "drums", "percussion",
}

# ── Word/phrase patterns for different suggestion types ──

_SECTION_MENTION_RE = re.compile(
    r"(?:in\s+the\s+|the\s+|your\s+)?"
    r"(?P<section>" + "|".join(re.escape(n) for n in sorted(_SECTION_NAMES, key=len, reverse=True)) + r")"
    r"(?:\s+(?:section|part|segment))?",
    re.IGNORECASE,
)

_VIBE_PATTERNS = (
    re.compile(
        r"(?:overall\s+)?(?:vibe|mood|feel|atmosphere|energy)\s*(?::|is|feels?)\s*(.+)$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:vibe|mood|atmosphere):\s*(.+)$", re.IGNORECASE),
)

_REFERENCE_PATTERNS = (
    re.compile(
        r"^(?:reference\s+track|reference|ref|similar\s+to|reminiscent\s+of|reminds\s+me\s+of|like)\s*(?::|–|-)?\s*(.+)$",
        re.IGNORECASE,
    ),
    re.compile(r"\"(.+?)\"\s*(?:vibe|style|reference)", re.IGNORECASE),
)

_ISSUE_PATTERNS = (
    re.compile(r"^(?:issue|problem|weakness|current\s+issue)\s*[:–\-]\s*(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?!\s*(?:issue|problem|weakness)\s*:)(.+\b(?:lacks contrast|feels flat|needs more movement|is muddy|is weak|needs work|doesn't work|not working|could be improved)\b.*)$",
        re.IGNORECASE,
    ),
)

_GOAL_PATTERNS = (
    re.compile(r"^(?:goal|aim|priority|focus|next\s+action)\s*[:–\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^((?:increase|improve|add|create|build|try|explore|experiment|strengthen|tighten|polish|refine)\s+.+)$", re.IGNORECASE),
)

_PROBLEM_PATTERNS = (
    re.compile(r"^(?:problem|focus|current\s+problem|main\s+challenge)\s*[:–\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^(?:note|remember|consider|try)\s*[:–\-]\s*(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?!\s*(?:issue|problem|weakness|goal|aim|priority|focus|current problem|note|remember|consider|try)\s*:)"
        r"(.+\bmay need\b.+)$",
        re.IGNORECASE,
    ),
)

_STAGES = (
    "idea", "sketch", "writing", "arrangement", "sound_design",
    "production", "mixing", "mastering", "finalizing",
)

_BPM_PATTERNS = (
    re.compile(r"\b(\d{2,3})\s*(?:bpm|beats per minute)\b", re.IGNORECASE),
    re.compile(r"\bbpm\s*[:–\-]?\s*(\d{2,3})\b", re.IGNORECASE),
    re.compile(r"\btempo\s*[:–\-]?\s*(\d{2,3})\b", re.IGNORECASE),
)
_KEY_RE = re.compile(
    r"(?:key\s+of\s+|\bkey\s*[:–\-]\s*|[\[\(])([A-Ga-g][#b]?(?:m|min|major|minor)?)",
    re.IGNORECASE,
)


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
            known_issues=self._extract_items(answer, _ISSUE_PATTERNS, existing=track_context.known_issues, limit=3),
            goals=self._extract_items(answer, _GOAL_PATTERNS, existing=track_context.goals, limit=3),
            current_stage=self._extract_stage(answer),
            current_problem=self._extract_problem(answer, existing=track_context.current_problem),
            vibe_suggestions=self._extract_items(answer, _VIBE_PATTERNS, existing=track_context.vibe, limit=3),
            reference_track_suggestions=self._extract_items(answer, _REFERENCE_PATTERNS, existing=track_context.reference_tracks, limit=3),
            section_suggestions=self._extract_section_info(answer),
            section_focus=self._extract_section_focus(answer, track_context),
            bpm_suggestion=self._extract_bpm(answer, track_context),
            key_suggestion=self._extract_key(answer, track_context),
        )

        if suggestions.current_stage and suggestions.current_stage == track_context.current_stage:
            suggestions.current_stage = None
        if suggestions.current_problem and suggestions.current_problem == track_context.current_problem:
            suggestions.current_problem = None
        if suggestions.vibe_suggestions:
            normalized_vibes = {v.strip().lower() for v in track_context.vibe}
            suggestions.vibe_suggestions = [
                v for v in suggestions.vibe_suggestions
                if v.strip().lower() not in normalized_vibes
            ][:3]
            if not suggestions.vibe_suggestions:
                suggestions.vibe_suggestions = []

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
            normalized_line = line.strip().lstrip("-* \u2022").strip()
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

    def _extract_section_info(
        self,
        answer: str,
    ) -> dict[str, dict[str, object]]:
        """Extract per-section issues/elements mentioned in the answer."""
        sections: dict[str, dict[str, object]] = {}
        current_section: str | None = None

        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped:
                current_section = None
                continue

            # Detect section heading
            section_match = _SECTION_MENTION_RE.search(stripped)
            if section_match:
                sec_name = section_match.group("section").lower().strip()
                if sec_name in _SECTION_NAMES:
                    current_section = sec_name
                    if sec_name not in sections:
                        sections[sec_name] = {"issues": [], "elements": [], "notes": ""}

            # Extract issues/elements from section context
            if current_section:
                section_data = sections[current_section]
                for ipattern in _ISSUE_PATTERNS:
                    imatch = ipattern.search(stripped)
                    if imatch:
                        issue = imatch.group(1).strip(" .:")
                        if issue and len(issue) > 5:
                            if issue not in section_data["issues"]:
                                section_data["issues"].append(issue)
                        break

                # Element detection within section
                if any(kw in stripped.lower() for kw in ("add ", "layer ", "introduce ", "bring in ")):
                    element = None
                    for prefix in ("add ", "layer ", "introduce ", "bring in "):
                        if stripped.lower().startswith(prefix):
                            element = stripped[len(prefix):].strip(" .:")
                            break
                    if element and element not in section_data["elements"]:
                        section_data["elements"].append(element)

        # Clean up empty sections
        return {
            k: v for k, v in sections.items()
            if v["issues"] or v["elements"]
        }

    def _extract_section_focus(
        self,
        answer: str,
        track_context: TrackContext,
    ) -> str | None:
        """Detect if the answer focuses on a particular section."""
        lowered = answer.lower()
        for section_name in sorted(_SECTION_NAMES, key=len, reverse=True):
            escaped = re.escape(section_name)
            section_patterns = (
                rf"\bfocus\s+(?:on|is)\s+(?:the\s+)?{escaped}\b",
                rf"\b(?:on|in|for)\s+(?:the\s+)?{escaped}\b",
                rf"\b(?:the|this|that)\s+{escaped}\b",
                rf"(?:^|\n)\s*{escaped}\s+(?:needs|section|is|feels)\b",
            )
            if any(re.search(pattern, lowered) for pattern in section_patterns):
                return section_name
        for section_name in track_context.sections:
            if section_name.lower() in lowered:
                return section_name
        return None

    def _extract_bpm(
        self,
        answer: str,
        track_context: TrackContext,
    ) -> int | None:
        """Extract a suggested BPM if mentioned and different from current."""
        suggested = None
        for pattern in _BPM_PATTERNS:
            bpm_match = pattern.search(answer)
            if bpm_match:
                suggested = int(bpm_match.group(1))
                break
        if suggested is None or not 20 <= suggested <= 300:
            return None
        if track_context.bpm and suggested == track_context.bpm:
            return None
        return suggested

    def _extract_key(
        self,
        answer: str,
        track_context: TrackContext,
    ) -> str | None:
        """Extract a suggested key if mentioned and different from current."""
        key_match = _KEY_RE.search(answer)
        if not key_match:
            return None
        raw = key_match.group(1).strip()
        suggested = raw[0].upper() + raw[1:] if raw else raw
        # Normalize to a common form
        suggested = suggested.replace("Major", "").replace("major", "").strip()
        if suggested.endswith("m"):
            pass  # Keep minor as-is (e.g. "Am")
        if track_context.key and suggested.lower() == track_context.key.lower():
            return None
        return suggested
