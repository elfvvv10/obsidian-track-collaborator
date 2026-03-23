"""Simple markdown chunking utilities."""

from __future__ import annotations

from pathlib import PurePosixPath
import re

from utils import Chunk, Note, compute_note_fingerprint, make_note_key, normalize_path, slugify


def chunk_notes(
    notes: list[Note],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Split notes into simple Markdown-aware chunks with overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []

    for note in notes:
        text = note.content.strip()
        if not text:
            continue

        normalized_text = text.replace("\r\n", "\n").strip()
        note_path = normalize_path(note.path)
        note_key = make_note_key(note_path)
        note_fingerprint = compute_note_fingerprint(note_path, normalized_text)
        source_dir = _path_directory(note_path)

        chunk_texts = _chunk_markdown_text(normalized_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_info in enumerate(chunk_texts):
            chunks.append(
                Chunk(
                    id=f"{slugify(note.path)}-{note_fingerprint[:12]}-{chunk_index}",
                    text=chunk_info["text"],
                    source_path=note_path,
                    note_title=note.title,
                    chunk_index=chunk_index,
                    source_dir=source_dir,
                    heading_context=chunk_info["heading_context"],
                    note_key=note_key,
                    note_fingerprint=note_fingerprint,
                )
            )

    return chunks


def _chunk_markdown_text(text: str, *, chunk_size: int, overlap: int) -> list[dict[str, str]]:
    sections = _split_into_sections(text)
    chunks: list[dict[str, str]] = []
    previous_text = ""

    for heading, section_text in sections:
        paragraphs = _split_large_section(section_text, chunk_size=chunk_size, heading=heading)
        buffer = ""

        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) <= chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(
                    {
                        "text": _apply_overlap(previous_text, buffer, overlap),
                        "heading_context": heading,
                    }
                )
                previous_text = buffer

            if len(paragraph) <= chunk_size:
                buffer = paragraph
            else:
                for fallback_text in _fallback_split(paragraph, chunk_size=chunk_size, overlap=overlap):
                    chunks.append(
                        {
                            "text": _apply_overlap(previous_text, fallback_text, overlap),
                            "heading_context": heading,
                        }
                    )
                    previous_text = fallback_text
                buffer = ""

        if buffer:
            chunks.append(
                {
                    "text": _apply_overlap(previous_text, buffer, overlap),
                    "heading_context": heading,
                }
            )
            previous_text = buffer

    return [chunk for chunk in chunks if chunk["text"].strip()]


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line.strip()):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = line.strip().lstrip("#").strip()
        current_lines.append(line.rstrip())

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections or [("", text)]


def _split_large_section(section_text: str, *, chunk_size: int, heading: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
    if not paragraphs:
        return []

    normalized_paragraphs: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        if index == 0 or not heading:
            normalized_paragraphs.append(paragraph)
            continue
        normalized_paragraphs.append(paragraph)
    return normalized_paragraphs


def _fallback_split(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    return [text[start : start + chunk_size].strip() for start in range(0, len(text), step) if text[start : start + chunk_size].strip()]


def _apply_overlap(previous_text: str, current_text: str, overlap: int) -> str:
    if not previous_text:
        return current_text.strip()

    overlap_text = previous_text[-overlap:].strip()
    if not overlap_text:
        return current_text.strip()
    return f"{overlap_text}\n\n{current_text.strip()}".strip()


def _path_directory(path: str) -> str:
    parent = str(PurePosixPath(path).parent)
    return "." if parent in {"", "."} else parent
