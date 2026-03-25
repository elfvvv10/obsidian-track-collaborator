"""Centralized provider-aware model client construction."""

from __future__ import annotations

from typing import TypeVar

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient, OpenAIChatClient
from model_clients import ChatModelClient, EmbeddingClient
from services.common import check_ollama_status


ChatClientType = TypeVar("ChatClientType", bound=ChatModelClient)
EmbeddingClientType = TypeVar("EmbeddingClientType", bound=EmbeddingClient)


def create_chat_client(
    config: AppConfig,
    *,
    model_override: str | None = None,
    client_cls: type[ChatClientType] | None = None,
) -> ChatModelClient:
    """Create the configured chat client while preserving test overrides."""
    if client_cls is not None:
        try:
            client = client_cls(config, model_override=model_override)
            if model_override:
                setattr(client, "model", model_override)
            return client
        except TypeError:
            client = client_cls(config)
            if model_override:
                setattr(client, "model", model_override)
            return client

    provider = config.chat_provider.strip().lower()
    if provider == "ollama":
        return OllamaChatClient(config, model_override=model_override)
    if provider == "openai":
        return OpenAIChatClient(config, model_override=model_override)
    raise ValueError(f"Unsupported chat provider: {config.chat_provider}")


def create_embedding_client(
    config: AppConfig,
    *,
    client_cls: type[EmbeddingClientType] | None = None,
) -> EmbeddingClient:
    """Create the configured embedding client while preserving test overrides."""
    if client_cls is not None:
        return client_cls(config)

    provider = config.embedding_provider.strip().lower()
    if provider == "ollama":
        return OllamaEmbeddingClient(config)
    if provider == "openai":
        raise NotImplementedError("The OpenAI embedding provider is not implemented yet.")
    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


def list_available_chat_models(config: AppConfig) -> tuple[list[str], str | None]:
    """Best-effort provider-aware helper for chat-model discovery."""
    provider = config.chat_provider.strip().lower()
    if provider == "ollama":
        try:
            client = OllamaChatClient(config)
            return client.list_available_models(), None
        except Exception as exc:
            return [], str(exc)
    if provider == "openai":
        if config.openai_chat_model.strip():
            return [config.openai_chat_model.strip()], None
        return [], "Set OPENAI_CHAT_MODEL to use the OpenAI chat provider."
    return [], f"Chat model discovery is not implemented for provider '{config.chat_provider}'."


def provider_status(config: AppConfig) -> tuple[bool | None, str]:
    """Return lightweight provider status information for UI readiness checks."""
    chat_provider = config.chat_provider.strip().lower()
    embedding_provider = config.embedding_provider.strip().lower()
    if chat_provider == "ollama" or embedding_provider == "ollama":
        return check_ollama_status(
            config.ollama_base_url,
            timeout_seconds=min(config.ollama_timeout_seconds, 5),
        )
    return None, (
        f"Provider status checks are not implemented for chat='{config.chat_provider}' "
        f"embedding='{config.embedding_provider}'."
    )
