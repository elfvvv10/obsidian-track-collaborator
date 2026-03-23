"""Simple markdown chunking utilities."""

from __future__ import annotations

from utils import Chunk, Note, slugify


def chunk_notes(
    notes: list[Note],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Split notes into overlapping character-based chunks."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for note in notes:
        text = note.content.strip()
        if not text:
            continue

        normalized = text.replace("\r\n", "\n")
        for chunk_index, start in enumerate(range(0, len(normalized), step)):
            chunk_text = normalized[start : start + chunk_size].strip()
            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    id=f"{slugify(note.path)}-{chunk_index}",
                    text=chunk_text,
                    source_path=note.path,
                    note_title=note.title,
                    chunk_index=chunk_index,
                )
            )

    return chunks
