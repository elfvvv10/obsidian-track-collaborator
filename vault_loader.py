"""Load markdown notes from an Obsidian vault."""

from __future__ import annotations

from pathlib import Path

from utils import Note


def load_notes(vault_path: Path) -> list[Note]:
    """Recursively load markdown notes from a vault directory."""
    notes: list[Note] = []

    for file_path in sorted(_iter_markdown_files(vault_path)):
        content = _read_text(file_path)
        if not content.strip():
            continue

        title = _extract_title(file_path, content)
        notes.append(
            Note(
                path=str(file_path.relative_to(vault_path)),
                title=title,
                content=content.strip(),
            )
        )

    return notes


def _iter_markdown_files(vault_path: Path):
    for path in vault_path.rglob("*.md"):
        if _should_skip(path, vault_path):
            continue
        if path.is_file():
            yield path


def _should_skip(path: Path, vault_path: Path) -> bool:
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
