"""Load markdown notes from an Obsidian vault."""

from __future__ import annotations

from pathlib import Path

from link_parser import extract_obsidian_links
from metadata_parser import extract_tags, parse_markdown_metadata
from utils import Note


def load_notes(vault_path: Path, excluded_paths: list[Path] | None = None) -> list[Note]:
    """Recursively load markdown notes from a vault directory."""
    notes: list[Note] = []
    excluded_paths = [path.resolve() for path in (excluded_paths or [])]

    for file_path in sorted(_iter_markdown_files(vault_path, excluded_paths)):
        raw_content = _read_text(file_path)
        frontmatter, content = parse_markdown_metadata(raw_content)
        if not content.strip():
            continue

        title = _extract_title(file_path, content)
        tags = extract_tags(frontmatter, content)
        links = extract_obsidian_links(content)
        notes.append(
            Note(
                path=str(file_path.relative_to(vault_path)),
                title=title,
                content=content.strip(),
                frontmatter=frontmatter,
                tags=tags,
                links=links,
            )
        )

    return notes


def _iter_markdown_files(vault_path: Path, excluded_paths: list[Path]):
    for path in vault_path.rglob("*.md"):
        if _should_skip(path, vault_path, excluded_paths):
            continue
        if path.is_file():
            yield path


def _should_skip(path: Path, vault_path: Path, excluded_paths: list[Path]) -> bool:
    resolved_path = path.resolve()
    for excluded_path in excluded_paths:
        if resolved_path == excluded_path or excluded_path in resolved_path.parents:
            return True

    relative_parts = path.relative_to(vault_path).parts
    return any(part.startswith(".") or part == ".obsidian" for part in relative_parts)


def _read_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8", errors="ignore")


def _extract_title(file_path: Path, content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or file_path.stem
    return file_path.stem
