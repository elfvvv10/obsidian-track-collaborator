"""Tests for CLI command behavior."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from config import AppConfig
from services.models import (
    IngestionResponse,
    QueryDebugInfo,
    QueryResponse,
    ResearchResponse,
    TrackContext,
    TrackContextUpdateProposal,
)
from utils import AnswerResult, RetrievedChunk


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


class CLITests(unittest.TestCase):
    def test_main_index_command_uses_incremental_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            with patch("main.load_config", return_value=config), patch(
                "main.run_index"
            ) as run_index_mock, patch("sys.argv", ["main.py", "index"]):
                exit_code = main.main()

            self.assertEqual(exit_code, 0)
            run_index_mock.assert_called_once_with(config, reset_store=False)

    def test_main_index_command_applies_chunk_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            with patch("main.load_config", return_value=config), patch(
                "main.run_index"
            ) as run_index_mock, patch(
                "sys.argv",
                [
                    "main.py",
                    "index",
                    "--chunk-size",
                    "800",
                    "--chunk-overlap",
                    "100",
                    "--chunking-strategy",
                    "sentence",
                ],
            ):
                exit_code = main.main()

            self.assertEqual(exit_code, 0)
            overridden_config = run_index_mock.call_args.args[0]
            self.assertEqual(overridden_config.chunk_size, 800)
            self.assertEqual(overridden_config.chunk_overlap, 100)
            self.assertEqual(overridden_config.chunking_strategy, "sentence")

    def test_main_ask_command_passes_filters_and_prints_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retrieved = [
                RetrievedChunk(
                    text="Agent note content",
                    metadata={
                        "note_title": "Agents",
                        "source_path": "projects/agents.md",
                        "heading_context": "Agents",
                    },
                    distance_or_score=0.1,
                )
            ]

            with patch("main.load_config", return_value=config), patch(
                "main.Retriever.retrieve", return_value=retrieved
            ) as retrieve_mock, patch(
                "main.OllamaChatClient.answer_with_prompt",
                return_value="Grounded answer",
            ), patch(
                "main.prompt_to_save",
                return_value=False,
            ), patch(
                "sys.argv",
                ["main.py", "ask", "What do my notes say?", "--folder", "projects", "--path-contains", "agents"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            retrieve_mock.assert_called_once()
            called_filters = retrieve_mock.call_args.kwargs["filters"]
            self.assertEqual(called_filters.folder, "projects")
            self.assertEqual(called_filters.path_contains, "agents")
            output = buffer.getvalue()
            self.assertIn("Grounded answer", output)
            self.assertIn("projects/agents.md", output)

    def test_main_ask_command_passes_retrieval_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retrieved = [
                RetrievedChunk(
                    text="Agent note content",
                    metadata={"note_title": "Agents", "source_path": "projects/agents.md"},
                    distance_or_score=0.1,
                )
            ]

            with patch("main.load_config", return_value=config), patch(
                "main.Retriever.retrieve", return_value=retrieved
            ) as retrieve_mock, patch(
                "main.OllamaChatClient.answer_with_prompt",
                return_value="Grounded answer",
            ), patch(
                "main.prompt_to_save",
                return_value=False,
            ), patch(
                "sys.argv",
                ["main.py", "ask", "What do my notes say?", "--top-k", "2", "--candidate-count", "4", "--rerank"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            called_options = retrieve_mock.call_args.kwargs["options"]
            self.assertEqual(called_options.top_k, 2)
            self.assertEqual(called_options.candidate_count, 4)
            self.assertTrue(called_options.rerank)

    def test_main_ask_command_passes_retrieval_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            with patch("main.load_config", return_value=config), patch(
                "main.QueryService.ask"
            ) as ask_mock, patch(
                "main.prompt_to_save",
                return_value=False,
            ), patch(
                "sys.argv",
                ["main.py", "ask", "What do my notes say?", "--retrieval-scope", "extended"],
            ):
                ask_mock.return_value = QueryResponse(
                    answer_result=AnswerResult(
                        answer="Grounded answer",
                        sources=["[Local 1] Agents (knowledge/agents.md)"],
                        retrieved_chunks=[
                            RetrievedChunk(
                                text="Agents note",
                                metadata={"note_title": "Agents", "source_path": "knowledge/agents.md"},
                                distance_or_score=0.1,
                            )
                        ],
                    ),
                )
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ask_mock.call_args.args[0]
            self.assertEqual(request.retrieval_scope.value, "extended")

    def test_main_ask_command_passes_track_context_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            with patch("main.load_config", return_value=config), patch(
                "main.QueryService.ask"
            ) as ask_mock, patch(
                "main.prompt_to_save",
                return_value=False,
            ), patch(
                "sys.argv",
                [
                    "main.py",
                    "ask",
                    "How do I improve this drop?",
                    "--track-id",
                    "moonlit_driver",
                    "--use-track-context",
                    "--section-focus",
                    "drop",
                ],
            ):
                ask_mock.return_value = QueryResponse(
                    answer_result=AnswerResult(
                        answer="Grounded answer",
                        sources=["[Local 1] Track Note (track.md)"],
                        retrieved_chunks=[
                            RetrievedChunk(
                                text="Track note",
                                metadata={"note_title": "Track Note", "source_path": "track.md"},
                                distance_or_score=0.1,
                            )
                        ],
                    ),
                )
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ask_mock.call_args.args[0]
            self.assertEqual(request.track_id, "moonlit_driver")
            self.assertTrue(request.use_track_context)
            self.assertEqual(request.section_focus, "drop")

    def test_run_ask_can_review_and_apply_track_context_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            response = QueryResponse(
                answer_result=AnswerResult(
                    answer="Try a simpler fill before the drop.",
                    sources=["[Local 1] Track Note (track.md)"],
                    retrieved_chunks=[
                        RetrievedChunk(
                            text="Track note",
                            metadata={"note_title": "Track Note", "source_path": "track.md"},
                            distance_or_score=0.1,
                        )
                    ],
                ),
                debug=QueryDebugInfo(active_section="drop"),
                track_context=TrackContext(track_id="moonlit_driver", track_name="Moonlit Driver"),
                track_context_update=TrackContextUpdateProposal(
                    track_id="moonlit_driver",
                    summary="Capture the next production move.",
                    add_to_lists={"next_actions": ["simplify the pre-drop fill"]},
                    section_focus="drop",
                    confidence="medium",
                ),
            )

            preview_context = TrackContext(
                track_id="moonlit_driver",
                track_name="Moonlit Driver",
                goals=["simplify the pre-drop fill"],
            )
            stub_service = unittest.mock.Mock()
            stub_service.ask.return_value = response
            stub_service.track_context_update_service.preview.return_value = preview_context
            stub_service.track_context_update_service.apply.return_value = preview_context
            stub_service.track_context_service.save.return_value = root / "output" / "track_contexts" / "moonlit_driver.yaml"

            with patch("main.QueryService", return_value=stub_service), patch(
                "builtins.input",
                side_effect=["y"],
            ), patch(
                "main.prompt_to_save",
                return_value=False,
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    main.run_ask(
                        config,
                        "How do I improve this drop?",
                        track_id="moonlit_driver",
                        use_track_context=True,
                        section_focus="drop",
                    )

            output = buffer.getvalue()
            self.assertIn("Suggested Track Context Update", output)
            self.assertIn("Updated Track Context Preview", output)
            self.assertIn("simplify the pre-drop fill", output)
            self.assertIn("Saved updated Track Context", output)
            stub_service.track_context_update_service.apply.assert_called_once()
            stub_service.track_context_service.save.assert_called_once_with(preview_context)

    def test_main_ingest_webpage_command_dispatches_to_ingestion_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            (config.curated_knowledge_path / "Arrangement").mkdir(parents=True)
            response = IngestionResponse(
                source="https://example.com/article",
                source_type="webpage",
                saved_path=root / "vault" / "ingested_webpages" / "article.md",
                title="Example Article",
                knowledge_category="Arrangement",
                index_triggered=True,
            )

            with patch("main.load_config", return_value=config), patch(
                "main.IngestionService.ingest_webpage",
                return_value=response,
            ) as ingest_mock, patch(
                "sys.argv",
                [
                    "main.py",
                    "ingest-webpage",
                    "https://example.com/article",
                    "--title",
                    "Example Article",
                    "--knowledge-category",
                    "arrangement",
                    "--index-now",
                ],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ingest_mock.call_args.args[0]
            self.assertEqual(request.source, "https://example.com/article")
            self.assertEqual(request.title_override, "Example Article")
            self.assertEqual(request.knowledge_category, "Arrangement")
            self.assertTrue(request.index_now)
            output = buffer.getvalue()
            self.assertIn("Ingestion Complete", output)
            self.assertIn("Example Article", output)
            self.assertIn("Knowledge Category: Arrangement", output)

    def test_main_ingest_youtube_command_dispatches_to_ingestion_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            (config.curated_knowledge_path / "References").mkdir(parents=True)
            response = IngestionResponse(
                source="https://www.youtube.com/watch?v=abc123xyz00",
                source_type="youtube",
                saved_path=root / "vault" / "ingested_youtube" / "video.md",
                title="Example Video",
                knowledge_category="References",
                index_triggered=True,
            )

            with patch("main.load_config", return_value=config), patch(
                "main.IngestionService.ingest_youtube",
                return_value=response,
            ) as ingest_mock, patch(
                "sys.argv",
                [
                    "main.py",
                    "ingest-youtube",
                    "https://www.youtube.com/watch?v=abc123xyz00",
                    "--title",
                    "Example Video",
                    "--knowledge-category",
                    "references",
                    "--index-now",
                ],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ingest_mock.call_args.args[0]
            self.assertEqual(request.source, "https://www.youtube.com/watch?v=abc123xyz00")
            self.assertEqual(request.title_override, "Example Video")
            self.assertEqual(request.knowledge_category, "References")
            self.assertTrue(request.index_now)
            output = buffer.getvalue()
            self.assertIn("Ingestion Complete", output)
            self.assertIn("Example Video", output)

    def test_main_ingest_pdf_command_dispatches_to_ingestion_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            (config.curated_knowledge_path / "Mixing").mkdir(parents=True)
            pdf_path = root / "source.pdf"
            response = IngestionResponse(
                source=str(pdf_path),
                source_type="pdf",
                saved_path=root / "vault" / "ingested_pdfs" / "note.md",
                title="Example PDF",
                knowledge_category="Mixing",
                index_triggered=True,
            )

            with patch("main.load_config", return_value=config), patch(
                "main.IngestionService.ingest_pdf",
                return_value=response,
            ) as ingest_mock, patch(
                "sys.argv",
                [
                    "main.py",
                    "ingest-pdf",
                    str(pdf_path),
                    "--title",
                    "Example PDF",
                    "--genre",
                    "Progressive House",
                    "--knowledge-category",
                    "mixing",
                    "--index-now",
                ],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ingest_mock.call_args.args[0]
            self.assertEqual(request.source, str(pdf_path))
            self.assertEqual(request.title_override, "Example PDF")
            self.assertEqual(request.import_genre, "Progressive House")
            self.assertEqual(request.knowledge_category, "Mixing")
            self.assertTrue(request.index_now)
            output = buffer.getvalue()
            self.assertIn("Ingestion Complete", output)
            self.assertIn("Example PDF", output)

    def test_main_ingest_docx_command_dispatches_to_ingestion_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            (config.curated_knowledge_path / "Sound Design").mkdir(parents=True)
            docx_path = root / "source.docx"
            response = IngestionResponse(
                source=str(docx_path),
                source_type="docx",
                saved_path=root / "vault" / "ingested_docx" / "note.md",
                title="Example DOCX",
                knowledge_category="Sound Design",
                index_triggered=True,
            )

            with patch("main.load_config", return_value=config), patch(
                "main.IngestionService.ingest_docx",
                return_value=response,
            ) as ingest_mock, patch(
                "sys.argv",
                [
                    "main.py",
                    "ingest-docx",
                    str(docx_path),
                    "--title",
                    "Example DOCX",
                    "--genre",
                    "Progressive House",
                    "--knowledge-category",
                    "sound design",
                    "--index-now",
                ],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ingest_mock.call_args.args[0]
            self.assertEqual(request.source, str(docx_path))
            self.assertEqual(request.title_override, "Example DOCX")
            self.assertEqual(request.import_genre, "Progressive House")
            self.assertEqual(request.knowledge_category, "Sound Design")
            self.assertTrue(request.index_now)
            output = buffer.getvalue()
            self.assertIn("Ingestion Complete", output)
            self.assertIn("Example DOCX", output)

    def test_main_ingest_pdf_command_rejects_unknown_knowledge_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            with patch("main.load_config", return_value=config), patch(
                "sys.argv",
                ["main.py", "ingest-pdf", str(root / "source.pdf"), "--knowledge-category", "Unknown"],
            ):
                exit_code = main.main()

            self.assertEqual(exit_code, 1)

    def test_main_research_command_dispatches_to_research_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            response = ResearchResponse(
                goal="Compare my notes with recent context",
                subquestions=["What do my notes say?", "What external context is relevant?"],
                steps=[],
                answer_result=AnswerResult(
                    answer="Final research answer",
                    sources=["[Local 1] Agents (agents.md)"],
                    retrieved_chunks=[],
                ),
            )

            with patch("main.load_config", return_value=config), patch(
                "main.ResearchService.research",
                return_value=response,
            ) as research_mock, patch(
                "main.prompt_to_save",
                return_value=False,
            ), patch(
                "sys.argv",
                ["main.py", "research", "Compare my notes with recent context", "--max-subquestions", "2"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = research_mock.call_args.args[0]
            self.assertEqual(request.goal, "Compare my notes with recent context")
            self.assertEqual(request.max_subquestions, 2)
            output = buffer.getvalue()
            self.assertIn("Research Plan", output)
            self.assertIn("Final research answer", output)
