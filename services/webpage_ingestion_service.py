"""Webpage-specific ingestion helpers."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re

import requests

from config import AppConfig
from services.import_genre_service import ImportGenreService
from services.ingestion_helpers import (
    build_ingested_markdown_note,
    fallback_title_from_url,
    make_ingestion_destination,
)
from services.knowledge_category_service import KnowledgeCategoryService
from services.models import IngestionRequest, IngestionResponse
from utils import ensure_directory


class WebpageIngestionService:
    """Fetch a webpage, extract readable text, and save it as a vault note."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.import_genre_service = ImportGenreService(config)
        self.knowledge_category_service = KnowledgeCategoryService(config)

    def ingest(self, request: IngestionRequest) -> IngestionResponse:
        """Import a webpage URL into the configured vault folder."""
        url = request.source.strip()
        if not url:
            raise ValueError("A webpage URL is required for ingestion.")

        html = self._fetch(url)
        extracted = _extract_webpage_content(html)
        title = request.title_override.strip() if request.title_override else extracted["title"]
        if not title:
            title = fallback_title_from_url(url, default_host="webpage")

        import_genre = self.import_genre_service.canonicalize(request.import_genre)
        knowledge_category = self.knowledge_category_service.validate_or_raise(request.knowledge_category)
        output_dir = self.import_genre_service.destination_for(
            self.config.webpage_ingestion_path,
            import_genre,
        )
        ensure_directory(output_dir)
        destination = make_ingestion_destination(output_dir, title)

        body = build_ingested_markdown_note(
            title=title,
            source_type="webpage_import",
            source_url=url,
            content_heading="Extracted Content",
            content=extracted["content"],
            status="imported",
            indexed=False,
            extra_frontmatter={
                "genre": import_genre,
                "knowledge_category": knowledge_category or "",
            },
            extra_metadata_lines=[
                ("Genre", import_genre),
                ("Knowledge Category", knowledge_category or ""),
            ],
        )
        destination.write_text(body, encoding="utf-8")

        return IngestionResponse(
            source=url,
            source_type="webpage",
            saved_path=destination,
            title=title,
            import_genre=import_genre,
            knowledge_category=knowledge_category,
            warnings=extracted["warnings"],
        )

    def _fetch(self, url: str) -> str:
        try:
            response = requests.get(
                url,
                timeout=self.config.webpage_fetch_timeout_seconds,
                headers={
                    "User-Agent": self.config.webpage_fetch_user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not fetch webpage content from {url}: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            raise RuntimeError(
                f"Webpage ingestion expected HTML content but received '{content_type or 'unknown'}'."
            )
        return response.text


def _extract_webpage_content(html: str) -> dict[str, object]:
    parser = _ReadableHTMLExtractor()
    parser.feed(html)
    parser.close()

    title = parser.title.strip()
    blocks = _dedupe_blocks(parser.blocks)
    warnings: list[str] = []
    if not blocks:
        warnings.append("Webpage text extraction returned very little readable content.")
    content = "\n\n".join(blocks).strip()
    if not content:
        content = "No readable content could be extracted from the page."
    return {
        "title": title,
        "content": content,
        "warnings": warnings,
    }


def _dedupe_blocks(blocks: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        normalized = " ".join(block.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


class _ReadableHTMLExtractor(HTMLParser):
    """Very small HTML-to-text extractor focused on readability over completeness."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.blocks: list[str] = []
        self._current_text: list[str] = []
        self._title_text: list[str] = []
        self._tag_stack: list[str] = []
        self._ignored_depth = 0
        self._ignored_tags = {"script", "style", "noscript", "svg", "header", "footer", "nav", "form"}
        self._block_tags = {"p", "div", "article", "section", "main", "li", "h1", "h2", "h3", "h4", "h5", "h6"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        self._tag_stack.append(lowered)
        if lowered in self._ignored_tags:
            self._flush_current_text()
            self._ignored_depth += 1
            return
        if lowered == "br":
            self._flush_current_text()

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._block_tags:
            self._flush_current_text()
        if lowered == "title":
            self.title = " ".join(self._title_text).strip()
        if lowered in self._ignored_tags and self._ignored_depth > 0:
            self._ignored_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        cleaned = _clean_text_fragment(data)
        if not cleaned:
            return
        if self._tag_stack and self._tag_stack[-1] == "title":
            self._title_text.append(cleaned)
            return
        self._current_text.append(cleaned)

    def close(self) -> None:
        self._flush_current_text()
        super().close()

    def _flush_current_text(self) -> None:
        if not self._current_text:
            return
        text = " ".join(self._current_text).strip()
        self._current_text = []
        if text:
            self.blocks.append(text)


def _clean_text_fragment(value: str) -> str:
    cleaned = unescape(value)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
