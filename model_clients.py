"""Provider-agnostic model client protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from services.prompt_service import PromptPayload


@runtime_checkable
class ChatModelClient(Protocol):
    """Minimal interface required by chat-generation workflows."""

    model: str

    def answer_with_prompt(self, prompt_payload: PromptPayload) -> str:
        """Answer using an already-built prompt payload."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Minimal interface required by retrieval and indexing flows."""

    model: str

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
