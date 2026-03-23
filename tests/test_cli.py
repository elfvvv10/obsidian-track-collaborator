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
from services.models import IngestionResponse
from utils import RetrievedChunk


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
                "main.OllamaChatClient.answer_question",
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
                "main.OllamaChatClient.answer_question",
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

    def test_main_ingest_webpage_command_dispatches_to_ingestion_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            response = IngestionResponse(
                source="https://example.com/article",
                source_type="webpage",
                saved_path=root / "vault" / "ingested_webpages" / "article.md",
                title="Example Article",
                index_triggered=True,
            )

            with patch("main.load_config", return_value=config), patch(
                "main.IngestionService.ingest_webpage",
                return_value=response,
            ) as ingest_mock, patch(
                "sys.argv",
                ["main.py", "ingest-webpage", "https://example.com/article", "--title", "Example Article", "--index-now"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ingest_mock.call_args.args[0]
            self.assertEqual(request.source, "https://example.com/article")
            self.assertEqual(request.title_override, "Example Article")
            self.assertTrue(request.index_now)
            output = buffer.getvalue()
            self.assertIn("Ingestion Complete", output)
            self.assertIn("Example Article", output)
