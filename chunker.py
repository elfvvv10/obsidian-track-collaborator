"""Simple markdown chunking utilities."""

from __future__ import annotations

from pathlib import PurePosixPath
import re

from services.arrangement_service import ArrangementService
from services.video_ingestion_service import parse_video_knowledge_document
from utils import Chunk, Note, compute_note_fingerprint, make_note_key, normalize_path, slugify


def chunk_notes(
    notes: list[Note],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
    strategy: str = "markdown",
) -> list[Chunk]:
    """Split notes into simple Markdown-aware chunks with overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []
    arrangement_service = ArrangementService()

    for note in notes:
        text = note.content.strip()
        if not text:
            continue

        normalized_text = text.replace("\r\n", "\n").strip()
        note_path = normalize_path(note.path)
        note_key = make_note_key(note_path)
        fingerprint_seed = "\n".join(
            [
                normalized_text,
                "|".join(note.tags),
                "|".join(note.linked_note_keys),
                note.source_kind,
                note.source_type,
                note.content_scope,
                note.content_category,
                note.import_genre,
                str((note.frontmatter or {}).get("track_name", "")).strip(),
                str((note.frontmatter or {}).get("genre", "")).strip(),
                str((note.frontmatter or {}).get("bpm", "")).strip(),
                str((note.frontmatter or {}).get("key", "")).strip(),
                str((note.frontmatter or {}).get("arrangement_version", "")).strip(),
            ]
        )
        note_fingerprint = compute_note_fingerprint(note_path, fingerprint_seed)
        source_dir = _path_directory(note_path)

        if arrangement_service.is_arrangement_document(note.frontmatter):
            chunks.extend(
                _chunk_arrangement_note(
                    note,
                    note_path=note_path,
                    note_key=note_key,
                    note_fingerprint=note_fingerprint,
                    source_dir=source_dir,
                    arrangement_service=arrangement_service,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
            continue

        if str(note.source_type).strip().lower() == "youtube_video":
            chunks.extend(
                _chunk_video_note(
                    note,
                    note_path=note_path,
                    note_key=note_key,
                    note_fingerprint=note_fingerprint,
                    source_dir=source_dir,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
            continue

        if strategy == "sentence":
            chunk_texts = _chunk_sentence_text(normalized_text, chunk_size=chunk_size, overlap=overlap)
        else:
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
                    tags=note.tags,
                    linked_note_keys=note.linked_note_keys,
                    source_kind=note.source_kind,
                    source_type=note.source_type,
                    content_scope=note.content_scope,
                    content_category=note.content_category,
                    import_genre=note.import_genre,
                )
            )

    return chunks


def _chunk_arrangement_note(
    note: Note,
    *,
    note_path: str,
    note_key: str,
    note_fingerprint: str,
    source_dir: str,
    arrangement_service: ArrangementService,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    arrangement = arrangement_service.parse(note.frontmatter or {}, note.content)
    texts_with_metadata: list[tuple[str, str, dict[str, object]]] = []

    overview_text = arrangement_service.render_overview_chunk(arrangement)
    if overview_text.strip():
        texts_with_metadata.append(
            (
                overview_text,
                "Arrangement Overview",
                {
                    "arrangement_track_name": arrangement.track_name or "",
                    "arrangement_genre": arrangement.genre or "",
                    "arrangement_section_id": "overview",
                    "arrangement_section_name": "Arrangement Overview",
                    "arrangement_energy": None,
                    "arrangement_version": (
                        str(arrangement.arrangement_version)
                        if arrangement.arrangement_version is not None
                        else ""
                    ),
                },
            )
        )

    for section in arrangement.sections:
        section_text = arrangement_service.render_section_chunk(arrangement, section)
        heading = f"{section.id} - {section.name}".strip()
        texts_with_metadata.append(
            (
                section_text,
                heading,
                {
                    "arrangement_track_name": arrangement.track_name or "",
                    "arrangement_genre": arrangement.genre or "",
                    "arrangement_section_id": section.id,
                    "arrangement_section_name": section.name,
                    "arrangement_energy": section.energy,
                    "arrangement_version": (
                        str(arrangement.arrangement_version)
                        if arrangement.arrangement_version is not None
                        else ""
                    ),
                },
            )
        )

    chunks: list[Chunk] = []
    chunk_counter = 0
    previous_text = ""
    for text, heading_context, metadata in texts_with_metadata:
        for chunk_info in _chunk_markdown_text(text, chunk_size=chunk_size, overlap=overlap):
            chunk_text = _apply_overlap(previous_text, chunk_info["text"], overlap)
            previous_text = chunk_info["text"]
            chunks.append(
                Chunk(
                    id=f"{slugify(note.path)}-{note_fingerprint[:12]}-{chunk_counter}",
                    text=chunk_text,
                    source_path=note_path,
                    note_title=note.title,
                    chunk_index=chunk_counter,
                    source_dir=source_dir,
                    heading_context=heading_context or chunk_info["heading_context"],
                    note_key=note_key,
                    note_fingerprint=note_fingerprint,
                    tags=note.tags,
                    linked_note_keys=note.linked_note_keys,
                    source_kind=note.source_kind,
                    source_type=note.source_type,
                    content_scope=note.content_scope,
                    content_category=note.content_category,
                    import_genre=note.import_genre,
                    arrangement_track_name=str(metadata["arrangement_track_name"]),
                    arrangement_genre=str(metadata["arrangement_genre"]),
                    arrangement_section_id=str(metadata["arrangement_section_id"]),
                    arrangement_section_name=str(metadata["arrangement_section_name"]),
                    arrangement_energy=metadata["arrangement_energy"],
                    arrangement_version=str(metadata["arrangement_version"]),
                )
            )
            chunk_counter += 1

    if chunks:
        return chunks

    return [
        Chunk(
            id=f"{slugify(note.path)}-{note_fingerprint[:12]}-0",
            text=note.content.strip(),
            source_path=note_path,
            note_title=note.title,
            chunk_index=0,
            source_dir=source_dir,
            heading_context=note.title,
            note_key=note_key,
            note_fingerprint=note_fingerprint,
            tags=note.tags,
            linked_note_keys=note.linked_note_keys,
            source_kind=note.source_kind,
            source_type=note.source_type,
            content_scope=note.content_scope,
            content_category=note.content_category,
            import_genre=note.import_genre,
            arrangement_track_name=arrangement.track_name or "",
            arrangement_genre=arrangement.genre or "",
            arrangement_version=(
                str(arrangement.arrangement_version)
                if arrangement.arrangement_version is not None
                else ""
            ),
        )
    ]


def _chunk_video_note(
    note: Note,
    *,
    note_path: str,
    note_key: str,
    note_fingerprint: str,
    source_dir: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    video_document = parse_video_knowledge_document(note.frontmatter or {}, note.content)
    if not video_document.sections and not video_document.summary:
        return []

    chunk_mode = (
        str((note.frontmatter or {}).get("video_index_mode", "")).strip().lower()
        or "sections"
    )
    texts_with_metadata: list[tuple[str, str, dict[str, object]]] = []

    overview_parts = ["# Video Knowledge Import"]
    if video_document.summary:
        overview_parts.extend(["", "## Summary", video_document.summary])
    if video_document.key_takeaways:
        overview_parts.extend(["", "## Key Takeaways", *[f"- {item}" for item in video_document.key_takeaways]])
    if video_document.topics:
        overview_parts.extend(["", f"Topics: {', '.join(video_document.topics)}"])
    overview_text = "\n".join(overview_parts).strip()
    if overview_text:
        texts_with_metadata.append(
            (
                overview_text,
                "Video Knowledge Import",
                {
                    "video_section_title": "Overview",
                    "video_start_time": "",
                    "video_end_time": "",
                    "video_chunk_kind": "summary",
                },
            )
        )

    for section in video_document.sections:
        heading = f"{section.title} [{int(section.start_time)}-{int(section.end_time)}]"
        summary_text = (
            f"# {section.title}\n\n"
            f"Timestamp: {section.start_time:.0f}s - {section.end_time:.0f}s\n"
            f"Summary: {section.summary}\n"
        ).strip()
        if section.key_points:
            summary_text += "\nKey Points:\n" + "\n".join(f"- {point}" for point in section.key_points)
        texts_with_metadata.append(
            (
                summary_text,
                heading,
                {
                    "video_section_title": section.title,
                    "video_start_time": section.start_time,
                    "video_end_time": section.end_time,
                    "video_chunk_kind": "section_summary",
                },
            )
        )

        if section.content:
            if chunk_mode in {"transcript", "both"}:
                content_chunks = _chunk_markdown_text(section.content, chunk_size=chunk_size, overlap=overlap)
                for content_index, chunk_info in enumerate(content_chunks, start=1):
                    texts_with_metadata.append(
                        (
                            chunk_info["text"],
                            f"{heading} | Content {content_index}",
                            {
                                "video_section_title": section.title,
                                "video_start_time": section.start_time,
                                "video_end_time": section.end_time,
                                "video_chunk_kind": "transcript",
                            },
                        )
                    )
            else:
                section_text = (
                    f"# {section.title}\n\n"
                    f"Timestamp: {section.start_time:.0f}s - {section.end_time:.0f}s\n"
                    f"Summary: {section.summary}\n\n"
                    f"Content:\n{section.content}\n\n"
                    f"Keywords: {', '.join(section.keywords)}"
                ).strip()
                texts_with_metadata.append(
                    (
                        section_text,
                        heading,
                        {
                            "video_section_title": section.title,
                            "video_start_time": section.start_time,
                            "video_end_time": section.end_time,
                            "video_chunk_kind": "section",
                        },
                    )
                )

    chunks: list[Chunk] = []
    chunk_counter = 0
    previous_text = ""
    for text, heading_context, metadata in texts_with_metadata:
        for chunk_info in _chunk_markdown_text(text, chunk_size=chunk_size, overlap=overlap):
            chunk_text = _apply_overlap(previous_text, chunk_info["text"], overlap)
            previous_text = chunk_info["text"]
            chunks.append(
                Chunk(
                    id=f"{slugify(note.path)}-{note_fingerprint[:12]}-{chunk_counter}",
                    text=chunk_text,
                    source_path=note_path,
                    note_title=note.title,
                    chunk_index=chunk_counter,
                    source_dir=source_dir,
                    heading_context=heading_context or chunk_info["heading_context"],
                    note_key=note_key,
                    note_fingerprint=note_fingerprint,
                    tags=note.tags,
                    linked_note_keys=note.linked_note_keys,
                    source_kind=note.source_kind,
                    source_type=note.source_type,
                    content_scope=note.content_scope,
                    content_category=note.content_category,
                    import_genre=note.import_genre,
                    video_title=video_document.video_title,
                    video_channel_name=video_document.channel_name or "",
                    video_source_url=video_document.source_url,
                    video_section_title=str(metadata["video_section_title"]),
                    video_start_time=str(metadata["video_start_time"]),
                    video_end_time=str(metadata["video_end_time"]),
                    video_duration_seconds=video_document.duration_seconds,
                    video_language=video_document.language or "",
                    video_schema_version=video_document.schema_version,
                    video_chunk_kind=str(metadata["video_chunk_kind"]),
                )
            )
            chunk_counter += 1

    return chunks


def _chunk_sentence_text(text: str, *, chunk_size: int, overlap: int) -> list[dict[str, str]]:
    sections = _split_into_sections(text)
    chunks: list[dict[str, str]] = []
    previous_text = ""

    for heading, section_text in sections:
        sentences = _split_into_sentences(section_text)
        if not sentences:
            continue

        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
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

            if len(sentence) <= chunk_size:
                buffer = sentence
            else:
                for fallback_text in _fallback_split(sentence, chunk_size=chunk_size, overlap=overlap):
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


def _split_into_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


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
