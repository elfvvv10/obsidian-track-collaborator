"""Shared helpers and lightweight data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
import re
from pathlib import Path


LOGGER_NAME = "obsidian_rag"


def get_logger() -> logging.Logger:
    """Return the shared console logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def current_timestamp() -> str:
    """Return a human-friendly local timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(value: str, max_length: int = 60) -> str:
    """Create a filesystem-friendly slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:max_length] or "note"


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_path(value: str) -> str:
    """Return a normalized, slash-separated path string."""
    return value.replace("\\", "/").strip("/")


def compute_content_hash(value: str) -> str:
    """Return a stable content hash."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compute_note_fingerprint(path: str, content: str) -> str:
    """Return a stable fingerprint for a note path and content."""
    normalized_path = normalize_path(path)
    return compute_content_hash(f"{normalized_path}\n{content}")


def make_note_key(path: str) -> str:
    """Return a stable key for note-level operations."""
    return compute_content_hash(normalize_path(path))


@dataclass(slots=True)
class Note:
    """A markdown note loaded from the Obsidian vault."""

    path: str
    title: str
    content: str
    frontmatter: dict[str, object] | None = None
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    linked_note_keys: tuple[str, ...] = ()


@dataclass(slots=True)
class Chunk:
    """A chunk of note content prepared for embedding."""

    id: str
    text: str
    source_path: str
    note_title: str
    chunk_index: int
    source_dir: str = "."
    heading_context: str = ""
    note_key: str = ""
    note_fingerprint: str = ""
    tags: tuple[str, ...] = ()
    linked_note_keys: tuple[str, ...] = ()


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned from the vector store."""

    text: str
    metadata: dict[str, object]
    distance_or_score: float | None = None


@dataclass(slots=True)
class AnswerResult:
    """The final answer and the supporting context used to build it."""

    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk]


@dataclass(slots=True)
class RetrievalFilters:
    """Optional filters applied during retrieval."""

    folder: str | None = None
    path_contains: str | None = None
    tag: str | None = None


@dataclass(slots=True)
class RetrievalOptions:
    """Optional retrieval controls for candidate selection and reranking."""

    top_k: int | None = None
    candidate_count: int | None = None
    rerank: bool | None = None
    boost_tags: tuple[str, ...] = ()
    include_linked_notes: bool | None = None
