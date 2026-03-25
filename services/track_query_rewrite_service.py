"""Retrieval-only query rewriting using Track Context."""

from __future__ import annotations

from services.models import TrackContext


class TrackQueryRewriteService:
    """Build compact retrieval queries using track metadata when available."""

    def rewrite(self, question: str, track_context: TrackContext | None) -> str:
        """Return a concise keyword-style retrieval query."""
        base_question = question.strip()
        if track_context is None or not base_question:
            return question

        parts: list[str] = [base_question]
        seen: set[str] = {base_question.lower()}

        def add(value: str | int | None) -> None:
            if value is None:
                return
            text = str(value).strip()
            if not text:
                return
            lowered = text.lower()
            joined = " ".join(parts).lower()
            if lowered in seen or lowered in joined or joined in lowered:
                return
            seen.add(lowered)
            parts.append(text)

        def add_many(values: list[str], *, limit: int | None = None) -> None:
            added = 0
            for value in values:
                before = len(parts)
                add(value)
                if len(parts) > before:
                    added += 1
                if limit is not None and added >= limit:
                    break

        add(track_context.genre)
        if track_context.workflow_mode != "general":
            add(track_context.workflow_mode)
        add(track_context.current_stage)
        add(track_context.current_section)
        add_many(track_context.known_issues, limit=2)
        add_many(track_context.goals, limit=2)
        add_many(track_context.vibe, limit=2)
        add(track_context.bpm)
        add(track_context.key)

        rewritten = " ".join(parts).strip()
        return rewritten or question
