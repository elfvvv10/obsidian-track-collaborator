"""Environment-based application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from utils import ensure_directory


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    obsidian_vault_path: Path
    obsidian_output_path: Path
    chroma_db_path: Path
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    top_k_results: int
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_candidate_multiplier: int = 2
    chunking_strategy: str = "markdown"
    enable_reranking: bool = False
    tag_boost_weight: float = 3.0
    enable_linked_note_expansion: bool = False
    max_linked_notes: int = 2
    linked_note_chunks_per_note: int = 1
    chroma_collection_name: str = "obsidian_notes"
    ollama_timeout_seconds: int = 60


def load_config() -> AppConfig:
    """Load and validate configuration from a .env file and environment."""
    load_dotenv()

    vault_path = _required_path_env("OBSIDIAN_VAULT_PATH", must_exist=True, directory_only=True)
    output_path = _required_path_env("OBSIDIAN_OUTPUT_PATH", must_exist=False, directory_only=True)
    chroma_path = _required_path_env("CHROMA_DB_PATH", must_exist=False, directory_only=True)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "hermes3").strip()
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").strip()
    top_k_results = _required_int_env("TOP_K_RESULTS", default=3, minimum=1)
    chunk_size = _required_int_env("CHUNK_SIZE", default=1000, minimum=100)
    chunk_overlap = _required_int_env("CHUNK_OVERLAP", default=150, minimum=0)
    retrieval_candidate_multiplier = _required_int_env(
        "RETRIEVAL_CANDIDATE_MULTIPLIER",
        default=2,
        minimum=1,
    )
    chunking_strategy = _choice_env(
        "CHUNKING_STRATEGY",
        default="markdown",
        choices={"markdown", "sentence"},
    )
    enable_reranking = _bool_env("ENABLE_RERANKING", default=False)
    tag_boost_weight = _required_float_env("TAG_BOOST_WEIGHT", default=3.0, minimum=0.0)
    enable_linked_note_expansion = _bool_env("ENABLE_LINKED_NOTE_EXPANSION", default=False)
    max_linked_notes = _required_int_env("MAX_LINKED_NOTES", default=2, minimum=1)
    linked_note_chunks_per_note = _required_int_env("LINKED_NOTE_CHUNKS_PER_NOTE", default=1, minimum=1)

    ensure_directory(output_path)
    ensure_directory(chroma_path)

    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    return AppConfig(
        obsidian_vault_path=vault_path,
        obsidian_output_path=output_path,
        chroma_db_path=chroma_path,
        ollama_base_url=base_url,
        ollama_chat_model=chat_model,
        ollama_embedding_model=embedding_model,
        top_k_results=top_k_results,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_candidate_multiplier=retrieval_candidate_multiplier,
        chunking_strategy=chunking_strategy,
        enable_reranking=enable_reranking,
        tag_boost_weight=tag_boost_weight,
        enable_linked_note_expansion=enable_linked_note_expansion,
        max_linked_notes=max_linked_notes,
        linked_note_chunks_per_note=linked_note_chunks_per_note,
    )


def _required_path_env(name: str, *, must_exist: bool, directory_only: bool) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")

    path = Path(value).expanduser().resolve()
    if must_exist and not path.exists():
        raise ValueError(f"Path configured in {name} does not exist: {path}")
    if path.exists() and directory_only and not path.is_dir():
        raise ValueError(f"Path configured in {name} must be a directory: {path}")
    return path


def _required_int_env(name: str, *, default: int, minimum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Received: {raw_value}") from exc

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {value}")
    return value


def _bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name, str(default)).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value. Received: {raw_value}")


def _choice_env(name: str, *, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}. Received: {value}")
    return value


def _required_float_env(name: str, *, default: float, minimum: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number. Received: {raw_value}") from exc

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {value}")
    return value
