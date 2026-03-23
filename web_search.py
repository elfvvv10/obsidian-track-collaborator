"""Lightweight web search client for optional external evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from urllib.parse import quote, unquote, urlparse, parse_qs

import requests

from config import AppConfig


@dataclass(slots=True)
class WebSearchResult:
    """A single external web result used as optional evidence."""

    title: str
    url: str
    snippet: str
    source_type: str = "web"


class DuckDuckGoWebSearchClient:
    """Small HTTP client for DuckDuckGo's instant answer API."""

    def __init__(self, config: AppConfig) -> None:
        self.base_url = config.web_search_api_url or "https://api.duckduckgo.com"
        self.max_results = config.web_search_max_results
        self.timeout = config.web_search_timeout_seconds
        self.html_search_url = "https://html.duckduckgo.com/html/"

    def search(self, query: str) -> list[WebSearchResult]:
        """Return a small set of web results for the given query."""
        try:
            response = requests.get(
                self.base_url,
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "no_redirect": 1,
                    "skip_disambig": 1,
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": "obsidian-rag-assistant/1.0",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Web search is unavailable: {exc}") from exc

        try:
            payload = _parse_json_payload(response)
        except ValueError:
            return self._search_html_fallback(query, response)
        results: list[WebSearchResult] = []

        abstract_text = str(payload.get("AbstractText", "")).strip()
        abstract_url = str(payload.get("AbstractURL", "")).strip()
        heading = str(payload.get("Heading", "")).strip() or "DuckDuckGo Result"
        if abstract_text and abstract_url:
            results.append(
                WebSearchResult(
                    title=heading,
                    url=abstract_url,
                    snippet=abstract_text,
                )
            )

        for item in _flatten_related_topics(payload.get("RelatedTopics", [])):
            text = str(item.get("Text", "")).strip()
            url = str(item.get("FirstURL", "")).strip()
            if not text or not url:
                continue
            title = text.split(" - ", 1)[0].strip() or "Web Result"
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=text,
                )
            )
            if len(results) >= self.max_results:
                break

        deduped: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for result in results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            deduped.append(result)
            if len(deduped) >= self.max_results:
                break
        return deduped

    def _search_html_fallback(
        self,
        query: str,
        initial_response: requests.Response,
    ) -> list[WebSearchResult]:
        """Fallback to DuckDuckGo's HTML results page when the JSON endpoint fails."""
        try:
            response = requests.post(
                self.html_search_url,
                data={"q": query},
                headers={
                    "User-Agent": "obsidian-rag-assistant/1.0",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            content_type = initial_response.headers.get("content-type", "unknown")
            preview = _response_preview(initial_response)
            raise RuntimeError(
                "Web search returned an invalid response format and the HTML fallback also failed. "
                f"Initial content type: '{content_type}'. Response preview: {preview}. "
                f"Fallback error: {exc}"
            ) from exc

        results = _parse_html_results(response.text, max_results=self.max_results)
        if results:
            return results

        initial_content_type = initial_response.headers.get("content-type", "unknown")
        initial_preview = _response_preview(initial_response)
        fallback_preview = _response_preview(response)
        raise RuntimeError(
            "Web search returned an invalid response format and the HTML fallback returned no usable results. "
            f"Initial content type: '{initial_content_type}'. Initial response preview: {initial_preview}. "
            f"Fallback response preview: {fallback_preview}"
        )


class WikipediaWebSearchClient:
    """Small HTTP client for Wikipedia search results."""

    def __init__(self, config: AppConfig) -> None:
        self.base_url = config.web_search_api_url or "https://en.wikipedia.org/w/api.php"
        self.max_results = config.web_search_max_results
        self.timeout = config.web_search_timeout_seconds

    def search(self, query: str) -> list[WebSearchResult]:
        """Return a small set of external results from Wikipedia."""
        try:
            response = requests.get(
                self.base_url,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "utf8": 1,
                    "format": "json",
                    "srlimit": self.max_results,
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": "obsidian-rag-assistant/1.0",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Web search is unavailable: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            preview = _response_preview(response)
            raise RuntimeError(
                "Wikipedia search returned an invalid response format. "
                f"Response preview: {preview}"
            ) from exc

        query_payload = payload.get("query", {})
        search_results = query_payload.get("search", [])
        if not isinstance(search_results, list):
            return []

        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for item in search_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            snippet = _strip_html(str(item.get("snippet", "")).strip())
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                )
            )
            if len(results) >= self.max_results:
                break
        return results


def _flatten_related_topics(items: object) -> list[dict[str, object]]:
    if not isinstance(items, list):
        return []

    flattened: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "Topics" in item:
            flattened.extend(_flatten_related_topics(item.get("Topics")))
        else:
            flattened.append(item)
    return flattened


def _parse_json_payload(response: requests.Response) -> dict[str, object]:
    """Parse JSON or JSON-like provider responses."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except ValueError:
        pass

    text = response.text.strip()
    if not text:
        raise ValueError("empty response body")

    # Some endpoints return JSON with a javascript-ish content type, and some
    # wrap the payload in a callback. Strip the wrapper if present.
    callback_match = re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\((.*)\);?$", text, flags=re.DOTALL)
    if callback_match:
        text = callback_match.group(1).strip()

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("response payload was not a JSON object")
    return payload


def _response_preview(response: requests.Response, max_length: int = 180) -> str:
    """Return a compact preview of a failed provider response."""
    text = getattr(response, "text", "")
    if not isinstance(text, str) or not text.strip():
        return "<empty body>"
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _parse_html_results(html: str, *, max_results: int) -> list[WebSearchResult]:
    """Extract a small set of search results from DuckDuckGo's HTML results page."""
    results: list[WebSearchResult] = []
    seen_urls: set[str] = set()

    pattern = re.compile(
        r'<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(?P<snippet>.*?)</a>|'
        r'<div[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(?P<divsnippet>.*?)</div>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    snippet_values = []
    for anchor_snippet, div_snippet in snippets:
        snippet_values.append(_strip_html(anchor_snippet or div_snippet))

    for index, match in enumerate(pattern.finditer(html)):
        raw_href = unquote(match.group("href"))
        url = _extract_result_url(raw_href)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = _strip_html(match.group("title")) or "Web Result"
        snippet = snippet_values[index] if index < len(snippet_values) else ""
        results.append(
            WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
            )
        )
        if len(results) >= max_results:
            break

    return results


def _extract_result_url(raw_href: str) -> str:
    """Normalize DuckDuckGo result URLs and unwrap redirect links when present."""
    if raw_href.startswith("//"):
        return "https:" + raw_href
    if raw_href.startswith("http://") or raw_href.startswith("https://"):
        parsed = urlparse(raw_href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            uddg = parse_qs(parsed.query).get("uddg", [])
            if uddg:
                return unquote(uddg[0])
        return raw_href
    return ""


def _strip_html(value: str) -> str:
    """Remove simple HTML tags and normalize whitespace."""
    cleaned = re.sub(r"<[^>]+>", " ", value)
    return " ".join(cleaned.split())
