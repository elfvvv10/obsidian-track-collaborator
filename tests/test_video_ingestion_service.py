"""Tests for YouTube/video ingestion v2 helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chunker import chunk_notes
from config import AppConfig
from services.prompt_service import PromptService
from services.video_ingestion_service import (
    VideoIngestionService,
    format_timestamp,
    parse_video_knowledge_markdown,
    render_video_knowledge_markdown,
)
from services.models import AnswerMode, RetrievalMode, VideoKnowledgeDocument, VideoKnowledgeSection, VideoTranscriptSegment
from utils import Note, RetrievedChunk
from vector_store import VectorStore


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="deepseek",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
    )


class VideoIngestionServiceTests(unittest.TestCase):
    def test_semantic_sections_merge_segments_into_idea_sized_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = VideoIngestionService(make_config(Path(tmp_dir)))
            sections = service._build_semantic_sections(
                [
                    VideoTranscriptSegment("The bass carries the main groove and should stay controlled.", 0.0, 8.0),
                    VideoTranscriptSegment("Use saturation for character, not for extra low-end mud.", 8.5, 16.0),
                    VideoTranscriptSegment("After a longer pause the tutorial moves into arrangement energy.", 24.5, 36.0),
                ]
            )

            self.assertEqual(len(sections), 2)
            self.assertIn("bass", sections[0].content.lower())
            self.assertGreaterEqual(sections[1].start_time, 24.0)

    def test_markdown_render_and_parse_round_trip(self) -> None:
        document = VideoKnowledgeDocument(
            source_url="https://www.youtube.com/watch?v=abc123xyz00",
            video_title="How to Build Better Basslines",
            channel_name="Producer Lab",
            published_at="2025-11-02",
            duration_seconds=1542,
            duration_readable="25:42",
            language="en",
            imported_at="2026-03-25 20:15:00",
            video_id="abc123xyz00",
            transcript_source="faster_whisper",
            whisper_model="small",
            section_count=2,
            transcript_chunk_count=5,
            import_genre="Techno",
            topics=["bass design", "groove"],
            tags=["youtube", "video_import", "techno"],
            summary="A practical walkthrough of bass role, groove, and low-end control.",
            key_takeaways=["Separate sub weight from moving mid-bass.", "Use saturation conservatively."],
            sections=[
                VideoKnowledgeSection(
                    title="Bass Role and Groove Framing",
                    start_time=0.0,
                    end_time=75.0,
                    summary="The speaker frames the bass as both rhythmic and tonal support.",
                    key_points=["Define the role of each bass layer."],
                    content="The speaker explains how a stable sub can support a more animated upper bass layer.",
                    keywords=["bass design", "groove"],
                )
            ],
            retrieval_notes=["Section summaries are synthesized from timestamped transcript segments."],
        )

        markdown = render_video_knowledge_markdown(document)
        parsed = parse_video_knowledge_markdown(markdown)

        self.assertIn('source_type: "youtube_video"', markdown)
        self.assertIn("## Sections", markdown)
        self.assertEqual(parsed.video_title, "How to Build Better Basslines")
        self.assertEqual(parsed.import_genre, "Techno")
        self.assertEqual(len(parsed.sections), 1)
        self.assertEqual(parsed.sections[0].title, "Bass Role and Groove Framing")

    def test_timestamp_format_is_zero_padded(self) -> None:
        self.assertEqual(format_timestamp(0), "00:00:00")
        self.assertEqual(format_timestamp(75), "00:01:15")
        self.assertEqual(format_timestamp(3723), "01:02:03")


class VideoChunkingTests(unittest.TestCase):
    def test_video_notes_chunk_into_summary_and_section_chunks(self) -> None:
        markdown = render_video_knowledge_markdown(
            VideoKnowledgeDocument(
                source_url="https://www.youtube.com/watch?v=abc123xyz00",
                video_title="Tripchain Tutorial Breakdown",
                channel_name="Producer Lab",
                imported_at="2026-03-25 20:15:00",
                section_count=1,
                transcript_chunk_count=2,
                summary="A focused breakdown of groove and bass decisions.",
                key_takeaways=["Use bass layers by role."],
                sections=[
                    VideoKnowledgeSection(
                        title="Bass Role and Groove Framing",
                        start_time=12.0,
                        end_time=88.0,
                        summary="The speaker explains bass role separation.",
                        key_points=["Separate sub support from animated character layers."],
                        content="The speaker explains how one bass layer can provide stable low-end weight while another handles movement.",
                        keywords=["bass design", "groove"],
                    )
                ],
            )
        )
        frontmatter, body = markdown.split("---\n", 2)[1:]
        note = Note(
            path="Imports/YouTube Imports/Generic/tripchain-tutorial.md",
            title="Tripchain Tutorial Breakdown",
            content=body.strip(),
            frontmatter={
                "source_type": "youtube_video",
                "video_title": "Tripchain Tutorial Breakdown",
                "source_url": "https://www.youtube.com/watch?v=abc123xyz00",
                "channel_name": "Producer Lab",
                "schema_version": "video_import_v1",
                "language": "en",
                "genre": "Generic",
            },
            source_kind="imported_content",
            source_type="youtube_video",
            content_scope="knowledge",
            content_category="imported_knowledge",
            import_genre="Generic",
        )

        chunks = chunk_notes([note], chunk_size=400, overlap=40)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(chunk.video_chunk_kind == "summary" for chunk in chunks))
        self.assertTrue(any(chunk.video_section_title == "Bass Role and Groove Framing" for chunk in chunks))

    def test_video_chunk_metadata_round_trips_into_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            store = VectorStore(config)
            store.reset()

            note = Note(
                path="Imports/YouTube Imports/Generic/video.md",
                title="Boris Brejcha Tutorial",
                content=(
                    "# Video Knowledge Import\n\n"
                    "## Summary\nA tutorial about groove and arrangement.\n\n"
                    "## Sections\n"
                    "### [00:00:12 - 00:01:28] Groove and Bass Role\n"
                    "**Summary:** The speaker explains bass role separation.\n\n"
                    "**Key points:**\n- Keep the sub stable.\n\n"
                    "**Content:**\nStable sub plus moving upper bass helps the groove feel deeper.\n\n"
                    "**Keywords:** groove, bass design\n"
                ),
                frontmatter={
                    "source_type": "youtube_video",
                    "video_title": "Boris Brejcha Tutorial",
                    "source_url": "https://www.youtube.com/watch?v=abc123xyz00",
                    "channel_name": "Producer Lab",
                    "duration_seconds": 1542,
                    "language": "en",
                    "schema_version": "video_import_v1",
                    "genre": "Generic",
                },
                source_kind="imported_content",
                source_type="youtube_video",
                content_scope="knowledge",
                content_category="imported_knowledge",
                import_genre="Generic",
            )
            chunks = chunk_notes([note], chunk_size=500, overlap=50)
            store.upsert_chunks(chunks, [[1.0, 0.0] for _ in chunks])
            rows = [metadata for _, metadata, _ in store.get_all_chunks() if metadata.get("source_type") == "youtube_video"]

            self.assertTrue(rows)
            self.assertTrue(any(row.get("video_title") == "Boris Brejcha Tutorial" for row in rows))
            self.assertTrue(any(row.get("video_section_title") == "Groove and Bass Role" for row in rows))


class VideoPromptTests(unittest.TestCase):
    def test_video_chunks_are_labeled_as_video_reference_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            payload = PromptService(make_config(root)).build_prompt_payload(
                "What does the Boris tutorial suggest?",
                [
                    RetrievedChunk(
                        text="Stable sub plus moving upper bass helps the groove feel deeper.",
                        metadata={
                            "note_title": "Boris Brejcha Tutorial",
                            "source_path": "Imports/YouTube Imports/Generic/boris.md",
                            "source_kind": "imported_content",
                            "source_type": "youtube_video",
                            "content_scope": "knowledge",
                            "content_category": "imported_knowledge",
                            "video_title": "Boris Brejcha Tutorial",
                            "video_section_title": "Groove and Bass Role",
                            "video_start_time": "12",
                            "video_end_time": "88",
                        },
                    )
                ],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
            )

            self.assertIn("Type: Video reference evidence", payload.user_prompt)
            self.assertIn("Video Section: Groove and Bass Role", payload.user_prompt)
            self.assertIn("Video Timestamp: 12 - 88", payload.user_prompt)
