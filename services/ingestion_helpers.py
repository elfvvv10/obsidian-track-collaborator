"""Shared helpers for external content ingestion services."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from utils import current_timestamp, ensure_directory, slugify


def build_ingested_markdown_note(
    *,
    title: str,
    source_type: str,
    source_url: str,
    content_heading: str,
    content: str,
    extra_frontmatter: dict[str, str] | None = None,
    extra_metadata_lines: list[tuple[str, str]] | None = None,
) -> str:
    """Build a simple, consistent markdown note for imported external content."""
    timestamp = current_timestamp()
    frontmatter_lines = [
        "---",
        f'title: "{escape_frontmatter(title)}"',
        f"source_type: {source_type}",
        f'source_url: "{escape_frontmatter(source_url)}"',
        f'ingested_at: "{timestamp}"',
    ]
    for key, value in (extra_frontmatter or {}).items():
        frontmatter_lines.append(f'{key}: "{escape_frontmatter(value)}"')
    frontmatter_lines.append("---")

    metadata_lines = [
        f"**Source URL:** {source_url}",
        f"**Ingested At:** {timestamp}",
    ]
    for label, value in (extra_metadata_lines or []):
        metadata_lines.append(f"**{label}:** {value}")

    return (
        "\n".join(frontmatter_lines)
        + "\n\n"
        + f"# {title}\n\n"
        + "\n\n".join(metadata_lines)
        + "\n\n"
        + f"## {content_heading}\n\n"
        + f"{content}\n"
    )


def make_ingestion_destination(output_dir: Path, title: str) -> Path:
    """Return a collision-safe destination path for an ingested note."""
    ensure_directory(output_dir)
    file_name = f"{current_timestamp().split(' ')[0]}-{slugify(title, max_length=50)}.md"
    return unique_destination(output_dir / file_name)


def unique_destination(path: Path) -> Path:
    """Return a unique file path by adding a deterministic numeric suffix when needed."""
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


def escape_frontmatter(value: str) -> str:
    """Escape double quotes for simple YAML frontmatter values."""
    return value.replace('"', '\\"')


def fallback_title_from_url(url: str, *, default_host: str = "external-content") -> str:
    """Generate a readable fallback title from a URL."""
    parsed = urlparse(url)
    host = parsed.netloc or default_host
    path = parsed.path.strip("/").replace("/", " ")
    if path:
        return f"{host} {path}".strip()
    return host
