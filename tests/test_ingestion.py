"""Tests for external webpage ingestion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from config import AppConfig
from services.ingestion_service import IngestionService
from services.models import IngestionRequest, IngestionResponse
from services.webpage_ingestion_service import WebpageIngestionService


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
                    IngestionRequest(source="https://example.com/article")
                )

            self.assertEqual(response.source_type, "webpage")
            self.assertEqual(response.title, "Test Article")
            self.assertTrue(response.saved_path.exists())
            self.assertIn(config.webpage_ingestion_folder, str(response.saved_path))

            content = response.saved_path.read_text(encoding="utf-8")
            self.assertIn('source_type: webpage', content)
            self.assertIn('source_url: "https://example.com/article"', content)
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
