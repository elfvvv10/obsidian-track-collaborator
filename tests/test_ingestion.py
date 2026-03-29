"""Tests for external webpage and document ingestion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import requests

from config import AppConfig
from services.docx_ingestion_service import DocxIngestionService
from services.ingestion_service import IngestionService
from services.models import IngestionRequest, IngestionResponse, VideoTranscriptSegment
from services.pdf_ingestion_service import PdfIngestionService
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
        curated_knowledge_folder="Knowledge",
    )


def write_minimal_pdf(path: Path, *, title: str = "", lines: list[str] | None = None) -> None:
    text_lines = lines or ["PDF line one.", "PDF line two."]
    title_fragment = f"/Title ({title})\n" if title else ""
    content_ops = " ".join(f"({line}) Tj" for line in text_lines)
    path.write_text(
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << >> >> endobj\n"
        f"4 0 obj << /Length {len(content_ops) + 32} >> stream\nBT /F1 12 Tf 72 220 Td {content_ops} ET\nendstream endobj\n"
        f"5 0 obj << {title_fragment}/Producer (Test) >> endobj\n"
        "trailer << /Root 1 0 R /Info 5 0 R >>\n%%EOF\n",
        encoding="latin-1",
    )


def write_minimal_docx(path: Path, *, title: str = "", paragraphs: list[str] | None = None) -> None:
    doc_paragraphs = paragraphs or ["DOCX line one.", "DOCX line two."]
    body = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        for paragraph in doc_paragraphs
    )
    core_title = f"<dc:title>{title}</dc:title>" if title else ""
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                '<Override PartName="/docProps/core.xml" '
                'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/>'
                '<Relationship Id="rId2" '
                'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
                'Target="docProps/core.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body>{body}</w:body></w:document>"
            ),
        )
        archive.writestr(
            "docProps/core.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<cp:coreProperties '
                'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/">'
                f"{core_title}</cp:coreProperties>"
            ),
        )


class WebpageIngestionTests(unittest.TestCase):
    def test_webpage_ingestion_saves_markdown_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()
            (config.curated_knowledge_path / "References").mkdir(parents=True)
            (config.curated_knowledge_path / "Arrangement").mkdir(parents=True)

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
                    IngestionRequest(
                        source="https://example.com/article",
                        import_genre="progressive house",
                        knowledge_category="arrangement",
                    )
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
            self.assertIn('created_by: "obsidian_track_collaborator"', content)
            self.assertIn('created_at: "', content)
            self.assertIn('source_url: "https://example.com/article"', content)
            self.assertIn('genre: "Progressive House"', content)
            self.assertIn('knowledge_category: "Arrangement"', content)
            self.assertIn("**Genre:** Progressive House", content)
            self.assertIn("**Knowledge Category:** Arrangement", content)
            self.assertIn("## Extracted Content", content)
            self.assertIn("This is the main content.", content)
            self.assertIn("# Test Article", content)
            self.assertEqual(response.knowledge_category, "Arrangement")

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
            self.assertIsNone(response.knowledge_category)
            content = response.saved_path.read_text(encoding="utf-8")
            self.assertNotIn("knowledge_category:", content)
            self.assertNotIn("**Knowledge Category:**", content)

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
            (config.curated_knowledge_path / "References").mkdir(parents=True)

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
                        knowledge_category="references",
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
            self.assertIn('created_by: "obsidian_track_collaborator"', content)
            self.assertIn('video_id: "abc123xyz00"', content)
            self.assertIn('genre: "New Groove"', content)
            self.assertIn('knowledge_category: "References"', content)
            self.assertIn("- **Knowledge Category:** References", content)
            self.assertIn("## Sections", content)
            self.assertIn("First transcript line.", content)
            self.assertIn("# Video Knowledge Import", content)
            self.assertIn("Producer Lab", content)
            self.assertEqual(response.section_count, 1)
            self.assertEqual(response.transcript_chunk_count, 2)
            self.assertEqual(response.knowledge_category, "References")

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


class PdfIngestionTests(unittest.TestCase):
    def test_pdf_ingestion_saves_markdown_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()
            (config.curated_knowledge_path / "Sound Design").mkdir(parents=True)
            pdf_path = root / "source.pdf"
            write_minimal_pdf(pdf_path, title="Synth Notes", lines=["Bass layer idea.", "Drop contrast tip."])

            response = PdfIngestionService(config).ingest(
                IngestionRequest(
                    source=str(pdf_path),
                    import_genre="progressive house",
                    knowledge_category="sound design",
                )
            )

            self.assertEqual(response.source_type, "pdf")
            self.assertEqual(response.title, "Synth Notes")
            self.assertTrue(response.saved_path.exists())
            self.assertIn(str(config.pdf_ingestion_path / "Progressive House"), str(response.saved_path))
            content = response.saved_path.read_text(encoding="utf-8")
            self.assertIn('source_type: "pdf_import"', content)
            self.assertIn('source_path: "', content)
            self.assertIn("source.pdf", content)
            self.assertIn('knowledge_category: "Sound Design"', content)
            self.assertIn("**Knowledge Category:** Sound Design", content)
            self.assertIn("## Extracted Content", content)
            self.assertIn("Bass layer idea.", content)
            self.assertEqual(response.knowledge_category, "Sound Design")

    def test_pdf_ingestion_uses_title_override_and_filename_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()
            pdf_path = root / "rough-arrangement.pdf"
            write_minimal_pdf(pdf_path, lines=["Arrangement note."])

            overridden = PdfIngestionService(config).ingest(
                IngestionRequest(source=str(pdf_path), title_override="Custom PDF Note")
            )
            fallback = PdfIngestionService(config).ingest(
                IngestionRequest(source=str(pdf_path))
            )

            self.assertEqual(overridden.title, "Custom PDF Note")
            self.assertEqual(fallback.title, "rough-arrangement")

    def test_pdf_ingestion_errors_on_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            with self.assertRaisesRegex(ValueError, "existing local file"):
                PdfIngestionService(config).ingest(IngestionRequest(source=str(root / "missing.pdf")))

    def test_ingestion_service_can_trigger_incremental_index_for_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            class StubPdfService:
                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def ingest(self, request: IngestionRequest) -> IngestionResponse:
                    return IngestionResponse(
                        source=request.source,
                        source_type="pdf",
                        saved_path=self.config.obsidian_vault_path / "ingested_pdfs" / "note.md",
                        title="Imported PDF",
                    )

            class StubIndexService:
                calls: list[bool] = []

                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def index(self, *, reset_store: bool) -> None:
                    self.calls.append(reset_store)

            response = IngestionService(
                config,
                pdf_service_cls=StubPdfService,
                index_service_cls=StubIndexService,
            ).ingest_pdf(
                IngestionRequest(source=str(root / "source.pdf"), index_now=True)
            )

            self.assertTrue(response.index_triggered)
            self.assertEqual(StubIndexService.calls, [False])


class DocxIngestionTests(unittest.TestCase):
    def test_docx_ingestion_saves_markdown_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()
            (config.curated_knowledge_path / "Arrangement").mkdir(parents=True)
            docx_path = root / "sound-design.docx"
            write_minimal_docx(docx_path, title="Sound Design Notes", paragraphs=["First paragraph.", "Second paragraph."])

            response = DocxIngestionService(config).ingest(
                IngestionRequest(
                    source=str(docx_path),
                    import_genre="Organic House",
                    knowledge_category="Arrangement",
                )
            )

            self.assertEqual(response.source_type, "docx")
            self.assertEqual(response.title, "Sound Design Notes")
            self.assertTrue(response.saved_path.exists())
            self.assertIn(str(config.docx_ingestion_path / "Organic House"), str(response.saved_path))
            content = response.saved_path.read_text(encoding="utf-8")
            self.assertIn('source_type: "docx_import"', content)
            self.assertIn('knowledge_category: "Arrangement"', content)
            self.assertIn("**Knowledge Category:** Arrangement", content)
            self.assertIn("First paragraph.", content)
            self.assertIn("Second paragraph.", content)
            self.assertEqual(response.knowledge_category, "Arrangement")

    def test_docx_ingestion_uses_title_override_and_filename_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()
            docx_path = root / "drop-notes.docx"
            write_minimal_docx(docx_path, paragraphs=["Drop note."])

            overridden = DocxIngestionService(config).ingest(
                IngestionRequest(source=str(docx_path), title_override="Custom DOCX Note")
            )
            fallback = DocxIngestionService(config).ingest(IngestionRequest(source=str(docx_path)))

            self.assertEqual(overridden.title, "Custom DOCX Note")
            self.assertEqual(fallback.title, "drop-notes")

    def test_docx_ingestion_errors_on_invalid_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()
            invalid_path = root / "legacy.doc"
            invalid_path.write_text("legacy", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "local .docx files"):
                DocxIngestionService(config).ingest(IngestionRequest(source=str(invalid_path)))

    def test_ingestion_service_can_trigger_incremental_index_for_docx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.obsidian_vault_path.mkdir()
            config.obsidian_output_path.mkdir()

            class StubDocxService:
                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def ingest(self, request: IngestionRequest) -> IngestionResponse:
                    return IngestionResponse(
                        source=request.source,
                        source_type="docx",
                        saved_path=self.config.obsidian_vault_path / "ingested_docx" / "note.md",
                        title="Imported DOCX",
                    )

            class StubIndexService:
                calls: list[bool] = []

                def __init__(self, config: AppConfig) -> None:
                    self.config = config

                def index(self, *, reset_store: bool) -> None:
                    self.calls.append(reset_store)

            response = IngestionService(
                config,
                docx_service_cls=StubDocxService,
                index_service_cls=StubIndexService,
            ).ingest_docx(
                IngestionRequest(source=str(root / "source.docx"), index_now=True)
            )

            self.assertTrue(response.index_triggered)
            self.assertEqual(StubIndexService.calls, [False])
