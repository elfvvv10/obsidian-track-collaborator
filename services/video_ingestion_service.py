"""Local-first video ingestion helpers for YouTube knowledge imports."""

from __future__ import annotations

from pathlib import Path
import re
import tempfile

import requests

from config import AppConfig
from metadata_parser import parse_markdown_metadata
from services.ingestion_helpers import escape_frontmatter, fallback_title_from_url, make_ingestion_destination
from services.models import (
    IngestionRequest,
    IngestionResponse,
    VideoKnowledgeDocument,
    VideoKnowledgeSection,
    VideoTranscriptSegment,
)
from utils import current_timestamp, ensure_directory

try:  # pragma: no cover - runtime dependency availability is covered by fallbacks
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - exercised via fallback path
    WhisperModel = None

try:  # pragma: no cover - runtime dependency availability is covered by fallbacks
    from yt_dlp import YoutubeDL
except ImportError:  # pragma: no cover - exercised via fallback path
    YoutubeDL = None

try:  # pragma: no cover - runtime dependency availability is covered by fallbacks
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:  # pragma: no cover - exercised via fallback path
    YouTubeTranscriptApi = None


VIDEO_IMPORT_TEMPLATE = """---
source_type: "youtube_video"
status: "imported"
indexed: false
created_by: "obsidian_track_collaborator"
imported_at: "2026-03-25 20:15:00"
video_title: "Full Video Title"
source_url: "https://www.youtube.com/watch?v=example"
platform: "youtube"
channel_name: "Channel Name"
published_at: "2025-11-02"
duration_seconds: 1542
duration_readable: "25:42"
language: "en"
content_type: "video_knowledge"
schema_version: "video_import_v1"
transcript_source: "faster_whisper"
whisper_model: "small"
retrieval_ready: true
section_count: 3
transcript_chunk_count: 9
topics:
  - bass design
  - groove
tags:
  - youtube
  - video_import
---

# Video Knowledge Import

## Source
- **Title:** Full Video Title
- **Channel:** Channel Name
- **URL:** https://www.youtube.com/watch?v=example
- **Published:** 2025-11-02
- **Duration:** 25:42
- **Language:** en
- **Imported:** 2026-03-25 20:15:00

## Summary
Short human-readable summary of the overall video.

## Key Takeaways
- Key lesson one.
- Key lesson two.

## Topics
- Bass design
- Groove

## Sections
### [00:00:00 - 00:02:10] Bass Role and Groove Framing
**Summary:** Short paragraph describing the main idea of this section.

**Key points:**
- Point one
- Point two

**Content:**
Cleaned section content that preserves the meaning of the source.

**Keywords:** bass design, groove, low end

## Retrieval Notes
- Section summaries are synthesized from timestamped transcript segments.
"""


class VideoIngestionService:
    """Build structured, retrieval-ready video knowledge notes."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def ingest_youtube(self, request: IngestionRequest, *, output_dir: Path, import_genre: str) -> IngestionResponse:
        """Ingest a YouTube URL into a structured video knowledge note."""
        url = request.source.strip()
        if not url:
            raise ValueError("A YouTube URL is required for ingestion.")

        video_id = extract_youtube_video_id(url)
        if not video_id:
            raise ValueError("Could not determine a YouTube video ID from the provided URL.")

        metadata = self._fetch_video_metadata(url, video_id)
        duration_seconds = metadata.get("duration_seconds")
        if duration_seconds and duration_seconds > self.config.youtube_max_duration_seconds:
            raise RuntimeError(
                f"Video duration {duration_seconds}s exceeds the configured limit of "
                f"{self.config.youtube_max_duration_seconds}s."
            )

        title = request.title_override.strip() if request.title_override else metadata.get("video_title", "").strip()
        if not title:
            title = fallback_title_from_url(url, default_host=f"youtube-{video_id}")

        transcript_segments, transcript_source, warnings = self._build_transcript(url, video_id)
        if not transcript_segments:
            raise RuntimeError("Video processing produced no usable transcript segments.")

        sections = self._build_semantic_sections(transcript_segments)
        if not sections:
            raise RuntimeError("Video processing produced no semantic sections.")

        document = VideoKnowledgeDocument(
            source_url=url,
            video_title=title,
            channel_name=_clean_str(metadata.get("channel_name")),
            published_at=_clean_str(metadata.get("published_at")),
            duration_seconds=duration_seconds,
            duration_readable=_format_duration(duration_seconds),
            language=_clean_str(metadata.get("language")),
            imported_at=current_timestamp(),
            video_id=video_id,
            transcript_source=transcript_source,
            whisper_model=self.config.youtube_whisper_model if transcript_source == "faster_whisper" else None,
            video_index_mode=self.config.youtube_index_mode,
            description_present=bool(metadata.get("description")),
            thumbnail_url=_clean_str(metadata.get("thumbnail_url")),
            section_count=len(sections),
            transcript_chunk_count=len(transcript_segments),
            domain_profile="electronic_music",
            workflow_type="knowledge_import",
            import_genre=import_genre,
            knowledge_category=request.knowledge_category,
            topics=self._build_topics(sections),
            tags=self._build_tags(import_genre, sections),
            summary=self._build_document_summary(sections),
            key_takeaways=self._build_key_takeaways(sections),
            sections=sections,
            retrieval_notes=[
                f"Section summaries are {'transcript-derived' if transcript_source == 'youtube_transcript_api' else 'heuristically synthesized'} from timestamped transcript segments.",
                "Timestamps reflect merged segment boundaries from ingestion-time transcript processing.",
                f"Index mode for this import was `{self.config.youtube_index_mode}`.",
            ],
        )

        if self.config.youtube_save_markdown_import_note:
            ensure_directory(output_dir)
            destination = make_ingestion_destination(output_dir, f"{title} Video Import")
            destination.write_text(render_video_knowledge_markdown(document), encoding="utf-8")
        else:
            raise RuntimeError("YOUTUBE_SAVE_MARKDOWN_IMPORT_NOTE=false is not supported in the current vault workflow.")

        return IngestionResponse(
            source=url,
            source_type="youtube",
            saved_path=destination,
            title=title,
            import_genre=import_genre,
            knowledge_category=request.knowledge_category,
            section_count=document.section_count,
            transcript_chunk_count=document.transcript_chunk_count,
            warnings=warnings,
        )

    def _fetch_video_metadata(self, url: str, video_id: str) -> dict[str, object]:
        if YoutubeDL is not None:
            try:
                with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                upload_date = _clean_str(info.get("upload_date"))
                return {
                    "video_title": _clean_str(info.get("title")) or f"YouTube {video_id}",
                    "channel_name": _clean_str(info.get("channel")),
                    "published_at": _format_upload_date(upload_date),
                    "duration_seconds": _coerce_int(info.get("duration")),
                    "language": _clean_str(info.get("language")),
                    "thumbnail_url": _clean_str(info.get("thumbnail")),
                    "description": _clean_str(info.get("description")),
                }
            except Exception:
                pass

        title = self._fetch_oembed_title(url, video_id)
        return {"video_title": title}

    def _fetch_oembed_title(self, url: str, video_id: str) -> str:
        try:
            response = requests.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=self.config.webpage_fetch_timeout_seconds,
                headers={"User-Agent": self.config.webpage_fetch_user_agent},
            )
            response.raise_for_status()
            data = response.json()
            title = _clean_str(data.get("title"))
            if title:
                return title
        except Exception:
            pass
        return f"YouTube {video_id}"

    def _build_transcript(self, url: str, video_id: str) -> tuple[list[VideoTranscriptSegment], str, list[str]]:
        warnings: list[str] = []

        if YoutubeDL is not None and WhisperModel is not None:
            try:
                segments = self._transcribe_with_faster_whisper(url)
                if segments:
                    return segments, "faster_whisper", warnings
            except Exception as exc:
                warnings.append(f"Local video transcription fallback used after audio/transcription failure: {exc}")

        if self.config.youtube_allow_transcript_fallback:
            segments = self._fetch_transcript_segments(video_id)
            if segments:
                warnings.append("Used youtube-transcript-api fallback instead of local audio transcription.")
                return segments, "youtube_transcript_api", warnings

        raise RuntimeError(
            "Could not extract usable video knowledge from this YouTube source. "
            "Install `yt-dlp` and `faster-whisper`, or enable transcript fallback."
        )

    def _transcribe_with_faster_whisper(self, url: str) -> list[VideoTranscriptSegment]:
        if YoutubeDL is None or WhisperModel is None:
            raise RuntimeError("Local video ingestion requires both yt-dlp and faster-whisper.")

        temp_root = Path(self.config.youtube_temp_dir).expanduser() if self.config.youtube_temp_dir else None
        if temp_root is not None:
            ensure_directory(temp_root)
        temp_manager = tempfile.TemporaryDirectory(dir=temp_root) if temp_root else tempfile.TemporaryDirectory()
        with temp_manager as tmp_dir:
            temp_path = Path(tmp_dir)
            output_template = str(temp_path / "source.%(ext)s")
            ydl_options = {
                "quiet": True,
                "no_warnings": True,
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "noplaylist": True,
            }
            with YoutubeDL(ydl_options) as ydl:
                ydl.download([url])

            audio_files = [path for path in temp_path.iterdir() if path.is_file()]
            if not audio_files:
                raise RuntimeError("Audio extraction did not produce a usable local media file.")
            audio_path = max(audio_files, key=lambda path: path.stat().st_size)

            model = WhisperModel(self.config.youtube_whisper_model, device="auto", compute_type="auto")
            segments, info = model.transcribe(
                str(audio_path),
                vad_filter=True,
                word_timestamps=False,
            )
            language = _clean_str(getattr(info, "language", ""))
            parsed_segments = [
                VideoTranscriptSegment(
                    text=text,
                    start_time=max(0.0, float(start)),
                    end_time=max(float(end), float(start)),
                )
                for text, start, end in (
                    (
                        _clean_transcript_text(getattr(segment, "text", "")),
                        getattr(segment, "start", 0.0),
                        getattr(segment, "end", getattr(segment, "start", 0.0)),
                    )
                    for segment in segments
                )
                if text
            ]
            if language and parsed_segments:
                return parsed_segments
            return parsed_segments

    def _fetch_transcript_segments(self, video_id: str) -> list[VideoTranscriptSegment]:
        if YouTubeTranscriptApi is None:
            return []

        transcript_items = _get_transcript_items(video_id)
        segments: list[VideoTranscriptSegment] = []
        for item in transcript_items:
            text = _clean_transcript_text(str(item.get("text", "")))
            if not text:
                continue
            start = _coerce_float(item.get("start")) or 0.0
            duration = _coerce_float(item.get("duration")) or 0.0
            end = start + duration if duration > 0 else start
            segments.append(
                VideoTranscriptSegment(
                    text=text,
                    start_time=max(0.0, start),
                    end_time=max(start, end),
                )
            )
        return segments

    def _build_semantic_sections(self, segments: list[VideoTranscriptSegment]) -> list[VideoKnowledgeSection]:
        if not segments:
            return []

        max_seconds = float(self.config.youtube_semantic_chunk_target_seconds)
        max_chars = int(self.config.youtube_semantic_chunk_target_chars)
        min_chars_for_boundary = max(250, int(max_chars * 0.55))
        gap_threshold = 4.0

        grouped: list[list[VideoTranscriptSegment]] = []
        current_group: list[VideoTranscriptSegment] = []

        for segment in segments:
            if not current_group:
                current_group = [segment]
                continue

            previous = current_group[-1]
            current_text = " ".join(item.text for item in current_group)
            current_duration = previous.end_time - current_group[0].start_time
            gap = max(0.0, segment.start_time - previous.end_time)
            should_split = False
            if gap >= gap_threshold:
                should_split = True
            elif len(current_text) >= max_chars:
                should_split = True
            elif current_duration >= max_seconds:
                should_split = True
            elif len(current_text) >= min_chars_for_boundary and _ends_with_sentence_boundary(previous.text):
                should_split = True

            if should_split:
                grouped.append(current_group)
                current_group = [segment]
            else:
                current_group.append(segment)

        if current_group:
            grouped.append(current_group)

        sections: list[VideoKnowledgeSection] = []
        for index, group in enumerate(grouped, start=1):
            content = _normalize_spacing(" ".join(segment.text for segment in group))
            if not content:
                continue
            keywords = _extract_keywords(content, limit=4)
            sections.append(
                VideoKnowledgeSection(
                    title=_build_section_title(content, keywords, index),
                    start_time=group[0].start_time,
                    end_time=group[-1].end_time,
                    summary=_build_section_summary(content),
                    key_points=_build_key_points(content),
                    content=content,
                    keywords=keywords,
                )
            )
        return sections

    def _build_topics(self, sections: list[VideoKnowledgeSection]) -> list[str]:
        return _dedupe_values(
            [keyword.replace("_", " ") for section in sections for keyword in section.keywords],
            limit=6,
        )

    def _build_tags(self, import_genre: str, sections: list[VideoKnowledgeSection]) -> list[str]:
        tags = ["youtube", "video_import"]
        if import_genre and import_genre.lower() != "generic":
            tags.append(import_genre.lower().replace(" ", "_"))
        tags.extend(keyword.lower().replace(" ", "_") for keyword in self._build_topics(sections)[:4])
        return _dedupe_values(tags, limit=8)

    def _build_document_summary(self, sections: list[VideoKnowledgeSection]) -> str:
        if not sections:
            return "No reliable summary could be extracted from the imported video."
        lead_sections = sections[:3]
        summaries = [section.summary.rstrip(".") for section in lead_sections if section.summary]
        if not summaries:
            return "No reliable summary could be extracted from the imported video."
        return ". ".join(summary for summary in summaries if summary).strip() + "."

    def _build_key_takeaways(self, sections: list[VideoKnowledgeSection]) -> list[str]:
        takeaways: list[str] = []
        for section in sections[:4]:
            for point in section.key_points[:2]:
                if point not in takeaways:
                    takeaways.append(point)
                if len(takeaways) >= 5:
                    return takeaways
        return takeaways


def render_video_knowledge_markdown(document: VideoKnowledgeDocument) -> str:
    """Render a canonical Obsidian markdown note for a video import."""
    frontmatter = [
        "---",
        f'source_type: "{escape_frontmatter(document.source_type)}"',
        f'status: "{escape_frontmatter(document.status)}"',
        f"indexed: {'true' if document.indexed else 'false'}",
        f'created_by: "{escape_frontmatter(document.created_by)}"',
        f'imported_at: "{escape_frontmatter(document.imported_at or current_timestamp())}"',
        f'video_title: "{escape_frontmatter(document.video_title)}"',
        f'source_url: "{escape_frontmatter(document.source_url)}"',
        f'platform: "{escape_frontmatter(document.platform)}"',
        f'channel_name: "{escape_frontmatter(document.channel_name or "")}"',
        f'published_at: "{escape_frontmatter(document.published_at or "")}"',
        f"duration_seconds: {document.duration_seconds if document.duration_seconds is not None else 0}",
        f'duration_readable: "{escape_frontmatter(document.duration_readable or "")}"',
        f'language: "{escape_frontmatter(document.language or "")}"',
        f'content_type: "{escape_frontmatter(document.content_type)}"',
        f'schema_version: "{escape_frontmatter(document.schema_version)}"',
    ]
    optional_scalar_fields = (
        ("transcript_source", document.transcript_source),
        ("whisper_model", document.whisper_model),
        ("video_index_mode", document.video_index_mode),
        ("video_id", document.video_id),
        ("thumbnail_url", document.thumbnail_url),
        ("import_notes", document.import_notes),
        ("domain_profile", document.domain_profile),
        ("workflow_type", document.workflow_type),
        ("genre", document.import_genre),
        ("knowledge_category", document.knowledge_category),
    )
    for key, value in optional_scalar_fields:
        if value is not None and str(value).strip():
            frontmatter.append(f'{key}: "{escape_frontmatter(str(value))}"')
    if document.description_present is not None:
        frontmatter.append(f"description_present: {'true' if document.description_present else 'false'}")
    frontmatter.append(f"retrieval_ready: {'true' if document.retrieval_ready else 'false'}")
    frontmatter.append(f"section_count: {document.section_count}")
    frontmatter.append(f"transcript_chunk_count: {document.transcript_chunk_count}")
    if document.topics:
        frontmatter.extend(["topics:", *[f"  - {escape_frontmatter(topic)}" for topic in document.topics]])
    if document.tags:
        frontmatter.extend(["tags:", *[f"  - {escape_frontmatter(tag)}" for tag in document.tags]])
    frontmatter.append("---")

    lines = [
        *frontmatter,
        "",
        "# Video Knowledge Import",
        "",
        "## Source",
        f"- **Title:** {document.video_title}",
        f"- **Channel:** {document.channel_name or 'Unknown'}",
        f"- **URL:** {document.source_url}",
        f"- **Published:** {document.published_at or 'Unknown'}",
        f"- **Duration:** {document.duration_readable or 'Unknown'}",
        f"- **Language:** {document.language or 'Unknown'}",
        *([f"- **Knowledge Category:** {document.knowledge_category}"] if document.knowledge_category else []),
        f"- **Imported:** {document.imported_at or current_timestamp()}",
        "",
        "## Summary",
        document.summary or "No reliable summary could be generated.",
        "",
        "## Key Takeaways",
    ]
    takeaways = document.key_takeaways or ["No high-confidence takeaways were extracted."]
    lines.extend(f"- {item}" for item in takeaways)
    lines.extend(["", "## Topics"])
    topics = document.topics or ["General"]
    lines.extend(f"- {item}" for item in topics)
    lines.extend(["", "## Sections"])

    for section in document.sections:
        lines.extend(
            [
                f"### [{format_timestamp(section.start_time)} - {format_timestamp(section.end_time)}] {section.title}",
                f"**Summary:** {section.summary or 'No section summary available.'}",
                "",
                "**Key points:**",
            ]
        )
        key_points = section.key_points or ["No high-confidence key points extracted."]
        lines.extend(f"- {point}" for point in key_points)
        lines.extend(
            [
                "",
                "**Content:**",
                section.content or "No section content available.",
                "",
                f"**Keywords:** {', '.join(section.keywords) if section.keywords else 'none'}",
                "",
            ]
        )

    if document.producer_notes:
        lines.extend(["## Producer Notes", *[f"- {note}" for note in document.producer_notes], ""])

    if document.retrieval_notes:
        lines.extend(["## Retrieval Notes", *[f"- {note}" for note in document.retrieval_notes], ""])

    return "\n".join(lines).rstrip() + "\n"


def parse_video_knowledge_markdown(raw_markdown: str) -> VideoKnowledgeDocument:
    """Parse a saved video knowledge note back into structured models."""
    frontmatter, body = parse_markdown_metadata(raw_markdown)
    return parse_video_knowledge_document(frontmatter, body)


def parse_video_knowledge_document(frontmatter: dict[str, object] | None, body: str) -> VideoKnowledgeDocument:
    """Parse frontmatter and body into a video knowledge document."""
    frontmatter = frontmatter or {}
    sections = _parse_video_sections(body)
    return VideoKnowledgeDocument(
        source_type=_clean_str(frontmatter.get("source_type")) or "youtube_video",
        source_url=_clean_str(frontmatter.get("source_url")) or "",
        video_title=_clean_str(frontmatter.get("video_title")) or "",
        platform=_clean_str(frontmatter.get("platform")) or "youtube",
        channel_name=_clean_str(frontmatter.get("channel_name")),
        published_at=_clean_str(frontmatter.get("published_at")),
        duration_seconds=_coerce_int(frontmatter.get("duration_seconds")),
        duration_readable=_clean_str(frontmatter.get("duration_readable")),
        language=_clean_str(frontmatter.get("language")),
        imported_at=_clean_str(frontmatter.get("imported_at")),
        schema_version=_clean_str(frontmatter.get("schema_version")) or "video_import_v1",
        content_type=_clean_str(frontmatter.get("content_type")) or "video_knowledge",
        status=_clean_str(frontmatter.get("status")) or "imported",
        indexed=str(frontmatter.get("indexed", "")).strip().lower() == "true",
        created_by=_clean_str(frontmatter.get("created_by")) or "obsidian_track_collaborator",
        video_id=_clean_str(frontmatter.get("video_id")),
        transcript_source=_clean_str(frontmatter.get("transcript_source")),
        whisper_model=_clean_str(frontmatter.get("whisper_model")),
        video_index_mode=_clean_str(frontmatter.get("video_index_mode")) or "sections",
        description_present=str(frontmatter.get("description_present", "")).strip().lower() == "true"
        if "description_present" in frontmatter
        else None,
        thumbnail_url=_clean_str(frontmatter.get("thumbnail_url")),
        retrieval_ready=str(frontmatter.get("retrieval_ready", "true")).strip().lower() != "false",
        section_count=_coerce_int(frontmatter.get("section_count")) or len(sections),
        transcript_chunk_count=_coerce_int(frontmatter.get("transcript_chunk_count")) or 0,
        domain_profile=_clean_str(frontmatter.get("domain_profile")),
        workflow_type=_clean_str(frontmatter.get("workflow_type")),
        import_notes=_clean_str(frontmatter.get("import_notes")),
        import_genre=_clean_str(frontmatter.get("genre")),
        knowledge_category=_clean_str(frontmatter.get("knowledge_category")),
        topics=_as_list(frontmatter.get("topics")),
        tags=_as_list(frontmatter.get("tags")),
        summary=_extract_heading_block(body, "Summary"),
        key_takeaways=_extract_bullets(_extract_heading_block(body, "Key Takeaways")),
        sections=sections,
        producer_notes=_extract_bullets(_extract_heading_block(body, "Producer Notes")),
        retrieval_notes=_extract_bullets(_extract_heading_block(body, "Retrieval Notes")),
    )


def extract_youtube_video_id(url: str) -> str | None:
    """Extract a supported YouTube video id from the given URL."""
    from urllib.parse import parse_qs, urlparse

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


def format_timestamp(seconds: float | int | None) -> str:
    """Return a zero-padded HH:MM:SS timestamp."""
    total_seconds = max(0, int(float(seconds or 0)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remainder = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:02d}"


def _build_section_summary(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text[:220].strip()
    summary = " ".join(sentences[:2]).strip()
    return summary[:280].rstrip(" ,;") + ("." if summary and summary[-1] not in ".!?" else "")


def _build_key_points(text: str) -> list[str]:
    sentences = _split_sentences(text)
    key_points: list[str] = []
    for sentence in sentences[:6]:
        cleaned = sentence.strip().rstrip(".")
        if len(cleaned.split()) < 5:
            continue
        key_points.append(cleaned[:160].rstrip(" ,;") + ".")
        if len(key_points) >= 3:
            break
    return key_points


def _build_section_title(text: str, keywords: list[str], index: int) -> str:
    if keywords:
        parts = [keyword.title() for keyword in keywords[:3]]
        return " and ".join(parts[:2]) if len(parts) >= 2 else parts[0]
    sentences = _split_sentences(text)
    if sentences:
        words = sentences[0].split()[:6]
        return " ".join(word.capitalize() for word in words).rstrip(".,")
    return f"Section {index}"


def _extract_keywords(text: str, *, limit: int) -> list[str]:
    stopwords = {
        "the", "and", "that", "with", "this", "from", "into", "they", "have", "about", "your", "just",
        "then", "there", "their", "what", "when", "where", "which", "because", "really", "video", "speaker",
        "like", "also", "will", "would", "could", "should", "using", "used", "make", "making", "made", "more",
        "less", "very", "much", "some", "into", "than", "them", "over", "under", "through", "being", "still",
    }
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
        if len(token) <= 3 or token in stopwords:
            continue
        counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [item[0].replace("_", " ") for item in ordered[:limit]]


def _parse_video_sections(body: str) -> list[VideoKnowledgeSection]:
    sections_block = _extract_heading_block(body, "Sections")
    pattern = re.compile(
        r"^### \[(?P<start>\d{2}:\d{2}:\d{2}) - (?P<end>\d{2}:\d{2}:\d{2})\] (?P<title>.+?)\n(?P<body>.*?)(?=^### |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    sections: list[VideoKnowledgeSection] = []
    for match in pattern.finditer(sections_block):
        section_body = match.group("body").strip()
        summary_match = re.search(r"\*\*Summary:\*\*\s*(.+)", section_body)
        keywords_match = re.search(r"\*\*Keywords:\*\*\s*(.+)", section_body)
        content_match = re.search(r"\*\*Content:\*\*\n(?P<content>.*?)(?=\n\*\*Keywords:\*\*|\Z)", section_body, re.DOTALL)
        key_points_block_match = re.search(
            r"\*\*Key points:\*\*\n(?P<points>.*?)(?=\n\*\*Content:\*\*|\Z)",
            section_body,
            re.DOTALL,
        )
        sections.append(
            VideoKnowledgeSection(
                title=match.group("title").strip(),
                start_time=_parse_timestamp(match.group("start")),
                end_time=_parse_timestamp(match.group("end")),
                summary=_clean_str(summary_match.group(1) if summary_match else "") or "",
                key_points=_extract_bullets(key_points_block_match.group("points") if key_points_block_match else ""),
                content=_normalize_spacing(content_match.group("content") if content_match else ""),
                keywords=[
                    keyword.strip()
                    for keyword in str(keywords_match.group(1) if keywords_match else "").split(",")
                    if keyword.strip() and keyword.strip().lower() != "none"
                ],
            )
        )
    return sections


def _extract_heading_block(body: str, heading: str) -> str:
    normalized_heading = heading.strip().lower()
    lines = body.splitlines()
    collected: list[str] = []
    in_block = False
    level = 0
    for raw_line in lines:
        stripped = raw_line.strip()
        match = re.match(r"^(#{2,6})\s+(.*)$", stripped)
        if match:
            current_level = len(match.group(1))
            current_heading = match.group(2).strip().lower()
            if in_block and current_level <= level:
                break
            if current_heading == normalized_heading:
                in_block = True
                level = current_level
                continue
        if in_block:
            collected.append(raw_line)
    return "\n".join(collected).strip()


def _extract_bullets(block: str) -> list[str]:
    return [line.strip()[2:].strip() for line in block.splitlines() if line.strip().startswith("- ")]


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _ends_with_sentence_boundary(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


def _clean_transcript_text(text: str) -> str:
    cleaned = _normalize_spacing(text)
    if cleaned in {"[Music]", "[Applause]", "[Laughter]"}:
        return ""
    return cleaned


def _normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _format_upload_date(value: str | None) -> str | None:
    if not value:
        return None
    if re.fullmatch(r"\d{8}", value):
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def _format_duration(duration_seconds: int | None) -> str | None:
    if duration_seconds is None:
        return None
    total = max(0, int(duration_seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _get_transcript_items(video_id: str) -> list[dict[str, object]]:
    if YouTubeTranscriptApi is None:
        return []
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return list(YouTubeTranscriptApi.get_transcript(video_id))
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)
    normalized: list[dict[str, object]] = []
    for item in fetched:
        if isinstance(item, dict):
            normalized.append(dict(item))
            continue
        normalized.append(
            {
                "text": getattr(item, "text", ""),
                "start": getattr(item, "start", 0.0),
                "duration": getattr(item, "duration", 0.0),
            }
        )
    return normalized


def _parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _coerce_int(value: object) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except ValueError:
        return None


def _coerce_float(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except ValueError:
        return None


def _clean_str(value: object) -> str | None:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or None


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    cleaned = str(value).strip()
    if not cleaned:
        return []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return [item.strip().strip("'\"") for item in cleaned[1:-1].split(",") if item.strip()]
    return [cleaned]


def _dedupe_values(values: list[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
        if len(deduped) >= limit:
            break
    return deduped
