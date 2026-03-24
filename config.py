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
    auto_save_answer: bool = False
    index_saved_answers: bool = False
    research_sessions_folder: str = "research_sessions"
    curated_knowledge_folder: str = "knowledge"
    index_research_sessions: bool = False
    index_webpage_imports: bool = False
    index_youtube_imports: bool = False
    web_search_provider: str = "wikipedia"
    web_search_api_url: str = ""
    web_search_max_results: int = 3
    web_search_timeout_seconds: int = 10
    webpage_ingestion_folder: str = "ingested_webpages"
    youtube_ingestion_folder: str = "ingested_youtube"
    auto_index_after_ingestion: bool = False
    webpage_fetch_timeout_seconds: int = 15
    webpage_fetch_user_agent: str = "obsidian-rag-assistant/1.0"
    track_critique_framework_path: str = ""
    chroma_collection_name: str = "obsidian_notes"
    ollama_timeout_seconds: int = 60

    @property
    def draft_answers_path(self) -> Path:
        """Return the path used for direct saved answers."""
        return self.obsidian_output_path

    @property
    def research_sessions_path(self) -> Path:
        """Return the path used for saved research workflow outputs."""
        return self.obsidian_vault_path / self.research_sessions_folder

    @property
    def curated_knowledge_path(self) -> Path:
        """Return the preferred curated-knowledge target folder inside the vault."""
        return self.obsidian_vault_path / self.curated_knowledge_folder

    @property
    def webpage_ingestion_path(self) -> Path:
        """Return the webpage-import folder path inside the vault."""
        return self.obsidian_vault_path / self.webpage_ingestion_folder

    @property
    def youtube_ingestion_path(self) -> Path:
        """Return the YouTube-import folder path inside the vault."""
        return self.obsidian_vault_path / self.youtube_ingestion_folder


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
    auto_save_answer = _bool_env("AUTO_SAVE_ANSWER", default=False)
    index_saved_answers = _bool_env("INDEX_SAVED_ANSWERS", default=False)
    research_sessions_folder = _relative_folder_env("RESEARCH_SESSIONS_FOLDER", default="research_sessions")
    curated_knowledge_folder = _relative_folder_env("CURATED_KNOWLEDGE_FOLDER", default="knowledge")
    index_research_sessions = _bool_env("INDEX_RESEARCH_SESSIONS", default=False)
    index_webpage_imports = _bool_env("INDEX_WEBPAGE_IMPORTS", default=False)
    index_youtube_imports = _bool_env("INDEX_YOUTUBE_IMPORTS", default=False)
    web_search_provider = _choice_env(
        "WEB_SEARCH_PROVIDER",
        default="wikipedia",
        choices={"wikipedia", "duckduckgo"},
    )
    web_search_api_url = os.getenv("WEB_SEARCH_API_URL", "").strip().rstrip("/")
    web_search_max_results = _required_int_env("WEB_SEARCH_MAX_RESULTS", default=3, minimum=1)
    web_search_timeout_seconds = _required_int_env("WEB_SEARCH_TIMEOUT_SECONDS", default=10, minimum=1)
    webpage_ingestion_folder = _relative_folder_env("WEBPAGE_INGESTION_FOLDER", default="ingested_webpages")
    youtube_ingestion_folder = _relative_folder_env("YOUTUBE_INGESTION_FOLDER", default="ingested_youtube")
    auto_index_after_ingestion = _bool_env("AUTO_INDEX_AFTER_INGESTION", default=False)
    webpage_fetch_timeout_seconds = _required_int_env("WEBPAGE_FETCH_TIMEOUT_SECONDS", default=15, minimum=1)
    webpage_fetch_user_agent = os.getenv(
        "WEBPAGE_FETCH_USER_AGENT",
        "obsidian-rag-assistant/1.0",
    ).strip()
    track_critique_framework_path = os.getenv("TRACK_CRITIQUE_FRAMEWORK_PATH", "").strip()

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
        auto_save_answer=auto_save_answer,
        index_saved_answers=index_saved_answers,
        research_sessions_folder=research_sessions_folder,
        curated_knowledge_folder=curated_knowledge_folder,
        index_research_sessions=index_research_sessions,
        index_webpage_imports=index_webpage_imports,
        index_youtube_imports=index_youtube_imports,
        web_search_provider=web_search_provider,
        web_search_api_url=web_search_api_url,
        web_search_max_results=web_search_max_results,
        web_search_timeout_seconds=web_search_timeout_seconds,
        webpage_ingestion_folder=webpage_ingestion_folder,
        youtube_ingestion_folder=youtube_ingestion_folder,
        auto_index_after_ingestion=auto_index_after_ingestion,
        webpage_fetch_timeout_seconds=webpage_fetch_timeout_seconds,
        webpage_fetch_user_agent=webpage_fetch_user_agent,
        track_critique_framework_path=track_critique_framework_path,
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


def _relative_folder_env(name: str, *, default: str) -> str:
    value = os.getenv(name, default).strip().replace("\\", "/").strip("/")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{name} must be a vault-relative folder path. Received: {value}")
    if ".." in path.parts:
        raise ValueError(f"{name} must stay inside the vault. Received: {value}")
    return value
