"""Local Ollama chat client and prompt helpers."""

from __future__ import annotations

from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException

from config import AppConfig
from utils import RetrievedChunk


SYSTEM_PROMPT = """You are a careful research assistant for an Obsidian vault.
Answer using the provided note context whenever possible.
If the context is weak, incomplete, or irrelevant, say that clearly instead of guessing.
Include source references using the note title or file path when they support the answer."""


class OllamaChatClient:
    """Small wrapper around the Ollama chat HTTP API."""

    def __init__(self, config: AppConfig) -> None:
        self.base_url = config.ollama_base_url
        self.model = config.ollama_chat_model
        self.timeout = config.ollama_timeout_seconds

    def answer_question(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Send a grounded chat prompt to Ollama and return the answer."""
        self._ensure_model_available()
        user_prompt = build_prompt(question, chunks)

        response = self._post_with_retry(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )

        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty chat response.")
        return content

    def _ensure_model_available(self) -> None:
        response = self._request("GET", "/api/tags")
        models = response.json().get("models", [])
        available_names = {item.get("name", "") for item in models}

        if self.model not in available_names and f"{self.model}:latest" not in available_names:
            raise RuntimeError(
                f"Chat model '{self.model}' is not installed in Ollama. "
                f"Run: ollama pull {self.model}"
            )

    def _post_with_retry(self, endpoint: str, json: dict[str, Any]) -> Response:
        last_error: Exception | None = None
        for _ in range(2):
            try:
                return self._request("POST", endpoint, json=json)
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(str(last_error)) from last_error

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> Response:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{endpoint}",
                timeout=self.timeout,
                **kwargs,
            )
        except RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama. Make sure Ollama is running and reachable at "
                f"{self.base_url}."
            ) from exc

        if response.status_code >= 400:
            raise RuntimeError(_format_ollama_error(response))
        return response


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build a grounded prompt from retrieved chunks and the user question."""
    if not chunks:
        context_block = "No relevant note context was retrieved."
    else:
        parts = []
        for index, chunk in enumerate(chunks, start=1):
            title = chunk.metadata.get("note_title", "Untitled note")
            source_path = chunk.metadata.get("source_path", "unknown")
            heading_context = chunk.metadata.get("heading_context", "")
            heading_line = f" | Section: {heading_context}" if heading_context else ""
            context_kind = "Linked note" if chunk.metadata.get("linked_context") else "Primary retrieval"
            tag_line = ""
            serialized_tags = chunk.metadata.get("tags_serialized", "")
            if isinstance(serialized_tags, str) and serialized_tags:
                tag_line = f"\nTags: {serialized_tags.replace('|', ', ')}"
            score_line = ""
            if chunk.distance_or_score is not None:
                score_line = f"\nRelevance distance: {chunk.distance_or_score:.4f}"
            parts.append(
                f"[Source {index}]\n"
                f"Type: {context_kind}\n"
                f"Title: {title}{heading_line}\n"
                f"Path: {source_path}{tag_line}{score_line}\n"
                f"Content:\n{chunk.text}"
            )
        context_block = "\n\n".join(parts)

    return (
        "Use the following retrieved note context to answer the question.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Respond with a concise, grounded answer. If the context is missing or insufficient, "
        "say so clearly."
    )


def _format_ollama_error(response: Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error_message = payload.get("error")
    if error_message:
        return f"Ollama request failed: {error_message}"
    return f"Ollama request failed with status {response.status_code}."
