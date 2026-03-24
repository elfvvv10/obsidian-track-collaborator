"""Helpers for keeping web evidence aligned to local retrieval topics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.models import RetrievalMode, WebQueryStrategy
from utils import RetrievedChunk
from web_search import WebSearchResult


@dataclass(slots=True)
class WebAlignmentResult:
    """Structured web-alignment output for query orchestration."""

    query: str
    strategy: WebQueryStrategy
    anchor_terms: tuple[str, ...]
    filtered_results: list[WebSearchResult]
    filtered_count: int = 0
    warning: str = ""


class WebAlignmentService:
    """Build local-guided web queries and filter clearly off-topic results."""

    _STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "your", "from", "into", "about",
        "what", "when", "where", "which", "while", "using", "used", "than", "them",
        "they", "their", "then", "have", "has", "had", "were", "will", "would",
        "could", "should", "also", "does", "into", "over", "under", "more", "less",
        "than", "notes", "note", "context", "recent", "external", "compare", "my",
    }

    def build_alignment(
        self,
        question: str,
        *,
        primary_chunks: list[RetrievedChunk],
        web_results: list[WebSearchResult],
        retrieval_mode: RetrievalMode,
        provider: str = "",
    ) -> WebAlignmentResult:
        """Return the web query strategy and filtered results."""
        web_query, strategy, anchor_terms = self.build_query(
            question,
            primary_chunks=primary_chunks,
            retrieval_mode=retrieval_mode,
            provider=provider,
        )

        filtered_results = self._filter_results(web_results, anchor_terms) if anchor_terms else list(web_results)
        filtered_count = max(0, len(web_results) - len(filtered_results))
        warning = ""
        if filtered_count and filtered_results:
            warning = (
                f"Web search was narrowed to local note topics and filtered out {filtered_count} off-topic result(s)."
            )
        elif filtered_count and not filtered_results:
            warning = "All external web results were discarded because they did not align with the local note topics."

        return WebAlignmentResult(
            query=web_query,
            strategy=strategy,
            anchor_terms=anchor_terms,
            filtered_results=filtered_results,
            filtered_count=filtered_count,
            warning=warning,
        )

    def build_query(
        self,
        question: str,
        *,
        primary_chunks: list[RetrievedChunk],
        retrieval_mode: RetrievalMode,
        provider: str = "",
    ) -> tuple[str, WebQueryStrategy, tuple[str, ...]]:
        """Build the web query and strategy from local anchors when available."""
        anchor_terms = self._extract_anchor_terms(primary_chunks)
        use_local_guidance = bool(anchor_terms) and retrieval_mode in {
            RetrievalMode.HYBRID,
            RetrievalMode.AUTO,
        }

        if use_local_guidance:
            strongest_anchor = self._extract_strongest_retry_anchor(primary_chunks)
            return (
                self._build_local_guided_query(
                    question,
                    anchor_terms,
                    provider=provider,
                    strongest_anchor=strongest_anchor,
                ),
                WebQueryStrategy.LOCAL_GUIDED,
                anchor_terms,
            )
        return question, WebQueryStrategy.RAW_QUESTION, anchor_terms

    def build_retry_query(
        self,
        question: str,
        *,
        primary_chunks: list[RetrievedChunk],
        provider: str = "",
    ) -> tuple[str, WebQueryStrategy, tuple[str, ...]]:
        """Build a lighter retry query using only the strongest local anchor."""
        anchor_terms = self._extract_anchor_terms(primary_chunks)
        strongest_anchor = self._extract_strongest_retry_anchor(primary_chunks)
        if not strongest_anchor:
            return question, WebQueryStrategy.RAW_QUESTION, anchor_terms

        if provider.strip().lower() == "wikipedia":
            retry_query = strongest_anchor
        else:
            retry_query = f'{question.strip()} "{strongest_anchor}"'.strip()
        return retry_query, WebQueryStrategy.LOCAL_GUIDED, (strongest_anchor.lower(),)

    def _build_local_guided_query(
        self,
        question: str,
        anchor_terms: tuple[str, ...],
        *,
        provider: str = "",
        strongest_anchor: str = "",
    ) -> str:
        provider_name = provider.strip().lower()
        if provider_name == "wikipedia" and strongest_anchor:
            return self._build_wikipedia_guided_query(question, strongest_anchor)

        query = question.strip()
        anchor_suffix = " ".join(anchor_terms[:6])
        if anchor_suffix:
            query = f"{query} {anchor_suffix}".strip()
        return query

    def _build_wikipedia_guided_query(self, question: str, strongest_anchor: str) -> str:
        question_lower = question.lower()
        if any(token in question_lower for token in {"recent", "latest", "current", "today", "this week"}):
            return f"{strongest_anchor} recent developments".strip()
        if "compare" in question_lower and any(token in question_lower for token in {"context", "external", "web"}):
            return f"{strongest_anchor} overview".strip()
        return strongest_anchor

    def _extract_anchor_terms(self, primary_chunks: list[RetrievedChunk]) -> tuple[str, ...]:
        ordered_terms: list[str] = []
        seen: set[str] = set()
        for chunk in primary_chunks[:3]:
            values = [
                str(chunk.metadata.get("note_title", "")),
                str(chunk.metadata.get("heading_context", "")),
                str(chunk.metadata.get("tags_serialized", "")).replace("|", " "),
            ]
            for value in values:
                for token in self._tokenize(value):
                    if token in self._STOPWORDS or len(token) < 3 or token in seen:
                        continue
                    seen.add(token)
                    ordered_terms.append(token)
        return tuple(ordered_terms)

    def _extract_strongest_retry_anchor(self, primary_chunks: list[RetrievedChunk]) -> str:
        for chunk in primary_chunks[:1]:
            for raw_value in (
                str(chunk.metadata.get("note_title", "")).strip(),
                str(chunk.metadata.get("heading_context", "")).strip(),
            ):
                cleaned = " ".join(self._tokenize(raw_value))
                if cleaned:
                    return cleaned
            tags = str(chunk.metadata.get("tags_serialized", "")).replace("|", " ").strip()
            cleaned_tags = " ".join(self._tokenize(tags))
            if cleaned_tags:
                return cleaned_tags.split()[0]
        return ""

    def _filter_results(
        self,
        web_results: list[WebSearchResult],
        anchor_terms: tuple[str, ...],
    ) -> list[WebSearchResult]:
        if not anchor_terms:
            return list(web_results)

        kept_results: list[WebSearchResult] = []
        anchor_set = set(anchor_terms)
        for result in web_results:
            haystack = " ".join([result.title, result.snippet, result.url])
            tokens = set(self._tokenize(haystack))
            meaningful_tokens = {token for token in tokens if token not in self._STOPWORDS and len(token) >= 3}
            if tokens & anchor_set:
                kept_results.append(result)
                continue
            # Keep sparse or generic results when we cannot confidently say they are off-topic.
            if len(meaningful_tokens) < 3:
                kept_results.append(result)
        return kept_results

    def _tokenize(self, value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value.lower())
