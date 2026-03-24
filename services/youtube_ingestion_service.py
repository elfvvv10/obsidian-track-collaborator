"""YouTube-specific ingestion helpers."""

from __future__ import annotations

from html import unescape
from urllib.parse import parse_qs, urlparse

import requests

from config import AppConfig
from services.ingestion_helpers import build_ingested_markdown_note, make_ingestion_destination
from services.models import IngestionRequest, IngestionResponse

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:  # pragma: no cover - exercised through runtime fallback
    YouTubeTranscriptApi = None


class YouTubeIngestionService:
    """Fetch a YouTube transcript and save it as a vault note."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def ingest(self, request: IngestionRequest) -> IngestionResponse:
        """Import a YouTube URL into the configured vault folder."""
        url = request.source.strip()
        if not url:
            raise ValueError("A YouTube URL is required for ingestion.")

        video_id = _extract_video_id(url)
        if not video_id:
            raise ValueError("Could not determine a YouTube video ID from the provided URL.")

        transcript = self._fetch_transcript(video_id)
        if not transcript.strip():
            raise RuntimeError("Transcript retrieval returned no usable content for this YouTube video.")

        title = request.title_override.strip() if request.title_override else self._fetch_title(url, video_id)
        output_dir = self.config.youtube_ingestion_path
        destination = make_ingestion_destination(output_dir, title)
        body = build_ingested_markdown_note(
            title=title,
            source_type="youtube_import",
            source_url=url,
            content_heading="Transcript",
            content=transcript,
            status="imported",
            indexed=False,
            extra_frontmatter={"youtube_video_id": video_id},
            extra_metadata_lines=[("Video ID", video_id)],
        )
        destination.write_text(body, encoding="utf-8")

        return IngestionResponse(
            source=url,
            source_type="youtube",
            saved_path=destination,
            title=title,
        )

    def _fetch_title(self, url: str, video_id: str) -> str:
        try:
            response = requests.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=self.config.webpage_fetch_timeout_seconds,
                headers={"User-Agent": self.config.webpage_fetch_user_agent},
            )
            response.raise_for_status()
            data = response.json()
            title = str(data.get("title", "")).strip()
            if title:
                return unescape(title)
        except requests.RequestException:
            pass
        except ValueError:
            pass
        return f"YouTube {video_id}"

    def _fetch_transcript(self, video_id: str) -> str:
        if YouTubeTranscriptApi is None:
            raise RuntimeError(
                "YouTube transcript ingestion requires the 'youtube-transcript-api' package to be installed."
            )

        try:
            transcript_items = _get_transcript_items(video_id)
        except Exception as exc:  # pragma: no cover - library-specific failure types
            raise RuntimeError(f"Could not retrieve a transcript for this YouTube video: {exc}") from exc

        lines = [str(item.get("text", "")).strip() for item in transcript_items]
        cleaned_lines = [line for line in lines if line and line not in {"[Music]", "[Applause]"}]
        return "\n\n".join(cleaned_lines).strip()


def _extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host in {"youtu.be", "www.youtu.be"}:
        return path.split("/")[0] or None

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if path == "watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if path.startswith("shorts/") or path.startswith("live/"):
            parts = path.split("/")
            if len(parts) >= 2:
                return parts[1] or None

    return None


def _get_transcript_items(video_id: str) -> list[dict[str, object]]:
    """Fetch transcript items across older and newer youtube-transcript-api versions."""
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return list(YouTubeTranscriptApi.get_transcript(video_id))

    api = YouTubeTranscriptApi()
    fetched_transcript = api.fetch(video_id)
    return [_normalize_transcript_item(item) for item in fetched_transcript]


def _normalize_transcript_item(item: object) -> dict[str, object]:
    """Normalize transcript snippets from dict-based and object-based library versions."""
    if isinstance(item, dict):
        return dict(item)

    text = getattr(item, "text", "")
    start = getattr(item, "start", None)
    duration = getattr(item, "duration", None)
    return {
        "text": text,
        "start": start,
        "duration": duration,
    }
