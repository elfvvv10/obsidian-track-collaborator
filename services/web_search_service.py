"""Service wrapper for optional external web search."""

from __future__ import annotations

from config import AppConfig
from web_search import DuckDuckGoWebSearchClient, WebSearchResult, WikipediaWebSearchClient


class WebSearchService:
    """Simple service wrapper around the configured web search client."""

    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        self.client = _build_web_search_client(config)

    def search(self, query: str) -> list[WebSearchResult]:
        """Return external web results for the question."""
        return self.client.search(query)


def _build_web_search_client(
    config: AppConfig,
) -> DuckDuckGoWebSearchClient | WikipediaWebSearchClient:
    if config.web_search_provider == "duckduckgo":
        return DuckDuckGoWebSearchClient(config)
    return WikipediaWebSearchClient(config)
