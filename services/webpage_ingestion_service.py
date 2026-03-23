"""Webpage-specific ingestion helpers."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlparse

import requests

from config import AppConfig
from services.models import IngestionRequest, IngestionResponse
from utils import current_timestamp, ensure_directory, slugify


class WebpageIngestionService:
    """Fetch a webpage, extract readable text, and save it as a vault note."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def ingest(self, request: IngestionRequest) -> IngestionResponse:
        """Import a webpage URL into the configured vault folder."""
        url = request.source.strip()
        if not url:
            raise ValueError("A webpage URL is required for ingestion.")

        html = self._fetch(url)
        extracted = _extract_webpage_content(html)
        title = request.title_override.strip() if request.title_override else extracted["title"]
        if not title:
            title = _fallback_title_from_url(url)

        output_dir = self.config.obsidian_vault_path / self.config.webpage_ingestion_folder
        ensure_directory(output_dir)
        file_name = f"{current_timestamp().split(' ')[0]}-{slugify(title, max_length=50)}.md"
        destination = _unique_destination(output_dir / file_name)

        body = _build_markdown_note(
            title=title,
            url=url,
            content=extracted["content"],
        )
        destination.write_text(body, encoding="utf-8")

        return IngestionResponse(
            source=url,
            source_type="webpage",
            saved_path=destination,
            title=title,
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


def _build_markdown_note(*, title: str, url: str, content: str) -> str:
    timestamp = current_timestamp()
    return (
        "---\n"
        f'title: "{_escape_frontmatter(title)}"\n'
        "source_type: webpage\n"
        f'source_url: "{_escape_frontmatter(url)}"\n'
        f'ingested_at: "{timestamp}"\n'
        "---\n\n"
        f"# {title}\n\n"
        f"**Source URL:** {url}\n\n"
        f"**Ingested At:** {timestamp}\n\n"
        "## Extracted Content\n\n"
        f"{content}\n"
    )


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _escape_frontmatter(value: str) -> str:
    return value.replace('"', '\\"')


def _fallback_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or "webpage"
    path = parsed.path.strip("/").replace("/", " ")
    if path:
        return f"{host} {path}".strip()
    return host


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
