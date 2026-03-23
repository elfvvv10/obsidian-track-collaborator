"""Local Ollama embedding client."""

from __future__ import annotations

from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException

from config import AppConfig


class OllamaEmbeddingClient:
    """Small wrapper around the Ollama embedding HTTP API."""

    def __init__(self, config: AppConfig) -> None:
        self.base_url = config.ollama_base_url
        self.model = config.ollama_embedding_model
        self.timeout = config.ollama_timeout_seconds

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        self._ensure_model_available()

        try:
            response = self._post_with_retry(
                "/api/embed",
                json={"model": self.model, "input": texts},
            )
            payload = response.json()
        except OllamaRequestError:
            # Older Ollama versions used /api/embeddings and only support one input.
            return [self._embed_single_with_legacy_endpoint(text) for text in texts]

        embeddings = payload.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings.")
        return embeddings

    def _embed_single_with_legacy_endpoint(self, text: str) -> list[float]:
        response = self._post_with_retry(
            "/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        payload = response.json()
        embedding = payload.get("embedding")
        if not embedding:
            raise RuntimeError("Ollama returned no embedding.")
        return embedding

    def _ensure_model_available(self) -> None:
        response = self._request("GET", "/api/tags")
        models = response.json().get("models", [])
        available_names = {item.get("name", "") for item in models}

        if self.model not in available_names and f"{self.model}:latest" not in available_names:
            raise RuntimeError(
                f"Embedding model '{self.model}' is not installed in Ollama. "
                f"Run: ollama pull {self.model}"
            )

    def _post_with_retry(self, endpoint: str, json: dict[str, Any]) -> Response:
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = self._request("POST", endpoint, json=json)
                return response
            except OllamaRequestError as exc:
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

        if response.status_code == 404:
            raise OllamaRequestError(f"Ollama endpoint not found: {endpoint}")
        if response.status_code >= 400:
            raise RuntimeError(_format_ollama_error(response))
        return response


class OllamaRequestError(RuntimeError):
    """Raised when an Ollama request fails in a retriable or compatibility-sensitive way."""


def _format_ollama_error(response: Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error_message = payload.get("error")
    if error_message:
        return f"Ollama request failed: {error_message}"
    return f"Ollama request failed with status {response.status_code}."
