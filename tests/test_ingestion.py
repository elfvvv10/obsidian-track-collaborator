"""Tests for external webpage ingestion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from config import AppConfig
from services.ingestion_service import IngestionService
from services.models import IngestionRequest, IngestionResponse, VideoTranscriptSegment
from services.video_ingestion_service import VideoIngestionService, _get_transcript_items
from services.webpage_ingestion_service import WebpageIngestionService
from services.youtube_ingestion_service import YouTubeIngestionService


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="hermes3",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
    )


class WebpageIngestionTests(unittest.TestCase):
    def test_webpage_ingestion_saves_markdown_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            class StubResponse:
                headers = {"content-type": "text/html; charset=utf-8"}
                text = """
                <html>
                  <head><title>Test Article</title></head>
                  <body>
                    <main>
                      <h1>Test Article</h1>
                      <p>This is the main content.</p>
                      <p>It should become markdown.</p>
                    </main>
                  </body>
                </html>
                """

                def raise_for_status(self) -> None:
                    return None

            with patch("services.webpage_ingestion_service.requests.get", return_value=StubResponse()):
                response = WebpageIngestionService(config).ingest(
                    IngestionRequest(source="https://example.com/article", import_genre="progressive house")
                )

            self.assertEqual(response.source_type, "webpage")
            self.assertEqual(response.title, "Test Article")
            self.assertTrue(response.saved_path.exists())
            self.assertIn(str(config.webpage_ingestion_path / "Progressive House"), str(response.saved_path))
            self.assertEqual(response.import_genre, "Progressive House")

            content = response.saved_path.read_text(encoding="utf-8")
            self.assertIn('source_type: "webpage_import"', content)
            self.assertIn('status: "imported"', content)
            self.assertIn("indexed: false", content)
            self.assertIn('created_by: "obsidian_rag_assistant"', content)
            self.assertIn('created_at: "', content)
            self.assertIn('source_url: "https://example.com/article"', content)
            self.assertIn('genre: "Progressive House"', content)
            self.assertIn("**Genre:** Progressive House", content)
            self.assertIn("## Extracted Content", content)
            self.assertIn("This is the main content.", content)
            self.assertIn("# Test Article", content)

    def test_webpage_ingestion_handles_fetch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            with patch(
                "services.webpage_ingestion_service.requests.get",
                side_effect=requests.RequestException("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Could not fetch webpage content"):
                    WebpageIngestionService(config).ingest(
                        IngestionRequest(source="https://example.com/article")
                    )

    def test_webpage_ingestion_defaults_to_generic_genre(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            class StubResponse:
                headers = {"content-type": "text/html; charset=utf-8"}
                text = "<html><head><title>Generic Article</title></head><body><p>Body</p></body></html>"

                def raise_for_status(self) -> None:
                    return None

            with patch("services.webpage_ingestion_service.requests.get", return_value=StubResponse()):
                response = WebpageIngestionService(config).ingest(
                    IngestionRequest(source="https://example.com/article")
                )

            self.assertIn(str(config.webpage_ingestion_path / "Generic"), str(response.saved_path))
            self.assertEqual(response.import_genre, "Generic")

    def test_webpage_ingestion_uses_title_override_and_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            class StubResponse:
                headers = {"content-type": "text/html"}
                text = "<html><head><title>Original Title</title></head><body><p>Body text.</p></body></html>"

                def raise_for_status(self) -> None:
                    return None

            with patch("services.webpage_ingestion_service.requests.get", return_value=StubResponse()):
                first = WebpageIngestionService(config).ingest(
                    IngestionRequest(source="https://example.com/article", title_override="Custom Note")
                )
                second = WebpageIngestionService(config).ingest(
                    IngestionRequest(source="https://example.com/article", title_override="Custom Note")
                )

            self.assertIn("custom-note", first.saved_path.name)
            self.assertTrue(second.saved_path.name.endswith("-2.md"))
            self.assertEqual(first.title, "Custom Note")
            self.assertEqual(second.title, "Custom Note")

    def test_ingestion_service_can_trigger_incremental_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            class StubWebpageService:
                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def ingest(self, request: IngestionRequest) -> IngestionResponse:
                    return IngestionResponse(
                        source=request.source,
                        source_type="webpage",
                        saved_path=self.config.obsidian_vault_path / "ingested_webpages" / "note.md",
                        title="Imported Page",
                    )

            class StubIndexService:
                calls: list[bool] = []

                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def index(self, *, reset_store: bool) -> None:
                    self.calls.append(reset_store)

            response = IngestionService(
                config,
                webpage_service_cls=StubWebpageService,
                index_service_cls=StubIndexService,
            ).ingest_webpage(
                IngestionRequest(source="https://example.com/article", index_now=True)
            )

            self.assertTrue(response.index_triggered)
            self.assertEqual(StubIndexService.calls, [False])

    def test_youtube_ingestion_saves_markdown_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            with patch.object(
                VideoIngestionService,
                "_fetch_video_metadata",
                return_value={
                    "video_title": "Test Video",
                    "channel_name": "Producer Lab",
                    "published_at": "2025-11-02",
                    "duration_seconds": 1542,
                    "language": "en",
                    "description": "A useful producer tutorial.",
                },
            ), patch.object(
                VideoIngestionService,
                "_build_transcript",
                return_value=(
                    [
                        VideoTranscriptSegment("First transcript line.", 0.0, 14.0),
                        VideoTranscriptSegment("Second transcript line with bass design details.", 14.0, 34.0),
                    ],
                    "faster_whisper",
                    [],
                ),
            ):
                response = YouTubeIngestionService(config).ingest(
                    IngestionRequest(
                        source="https://www.youtube.com/watch?v=abc123xyz00",
                        import_genre="New Groove",
                    )
                )

            self.assertEqual(response.source_type, "youtube")
            self.assertEqual(response.title, "Test Video")
            self.assertTrue(response.saved_path.exists())
            self.assertIn(str(config.youtube_ingestion_path / "New Groove"), str(response.saved_path))
            self.assertEqual(response.import_genre, "New Groove")

            content = response.saved_path.read_text(encoding="utf-8")
            self.assertIn('source_type: "youtube_video"', content)
            self.assertIn('status: "imported"', content)
            self.assertIn("indexed: false", content)
            self.assertIn('created_by: "obsidian_rag_assistant"', content)
            self.assertIn('video_id: "abc123xyz00"', content)
            self.assertIn('genre: "New Groove"', content)
            self.assertIn("## Sections", content)
            self.assertIn("First transcript line.", content)
            self.assertIn("# Video Knowledge Import", content)
            self.assertIn("Producer Lab", content)
            self.assertEqual(response.section_count, 1)
            self.assertEqual(response.transcript_chunk_count, 2)

    def test_youtube_ingestion_handles_video_processing_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            with patch.object(
                VideoIngestionService,
                "_fetch_video_metadata",
                return_value={"video_title": "Broken Video"},
            ), patch.object(
                VideoIngestionService,
                "_build_transcript",
                side_effect=RuntimeError("no transcript"),
            ):
                with self.assertRaisesRegex(RuntimeError, "no transcript"):
                    YouTubeIngestionService(config).ingest(
                        IngestionRequest(source="https://www.youtube.com/watch?v=abc123xyz00")
                    )

    def test_youtube_transcript_fetch_supports_instance_api_shape(self) -> None:
        class Snippet:
            def __init__(self, text: str, start: float = 0.0, duration: float = 1.0) -> None:
                self.text = text
                self.start = start
                self.duration = duration

        class StubTranscriptApi:
            def fetch(self, video_id: str):
                return [
                    Snippet("First line"),
                    Snippet("[Music]"),
                    Snippet("Second line"),
                ]

        with patch("services.video_ingestion_service.YouTubeTranscriptApi", StubTranscriptApi):
            transcript_items = _get_transcript_items("abc123xyz00")

        self.assertEqual(transcript_items[0]["text"], "First line")
        self.assertEqual(transcript_items[2]["text"], "Second line")

    def test_ingestion_service_can_trigger_incremental_index_for_youtube(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            class StubYouTubeService:
                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def ingest(self, request: IngestionRequest) -> IngestionResponse:
                    return IngestionResponse(
                        source=request.source,
                        source_type="youtube",
                        saved_path=self.config.obsidian_vault_path / "ingested_youtube" / "video.md",
                        title="Imported Video",
                    )

            class StubIndexService:
                calls: list[bool] = []

                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def index(self, *, reset_store: bool) -> None:
                    self.calls.append(reset_store)

            response = IngestionService(
                config,
                youtube_service_cls=StubYouTubeService,
                index_service_cls=StubIndexService,
            ).ingest_youtube(
                IngestionRequest(source="https://www.youtube.com/watch?v=abc123xyz00", index_now=True)
            )

            self.assertTrue(response.index_triggered)
            self.assertEqual(StubIndexService.calls, [False])
