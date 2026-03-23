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

    ensure_directory(output_path)
    ensure_directory(chroma_path)

    return AppConfig(
        obsidian_vault_path=vault_path,
        obsidian_output_path=output_path,
        chroma_db_path=chroma_path,
        ollama_base_url=base_url,
        ollama_chat_model=chat_model,
        ollama_embedding_model=embedding_model,
        top_k_results=top_k_results,
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
