"""Tests for Phase 4 save-back improvements."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import main
from config import AppConfig
from saver import save_answer
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
    )


class SaveTemplateTests(unittest.TestCase):
    def test_save_answer_uses_structured_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            result = AnswerResult(
                answer="Agents use tools to act.\n- They retrieve context.\n- They cite sources.",
                sources=["AI Agents (ai_agents.md)"],
                retrieved_chunks=[],
            )

            saved_path = save_answer(output_path, "What are AI agents?", result)
            contents = saved_path.read_text(encoding="utf-8")

            self.assertIn("## Summary", contents)
            self.assertIn("## Answer", contents)
            self.assertIn("## Key Points", contents)
            self.assertIn("## Sources", contents)
            self.assertIn("- They retrieve context.", contents)


class Phase4CLITests(unittest.TestCase):
    def test_main_ask_command_auto_saves_without_prompt_when_flag_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retrieved = [
                RetrievedChunk(
                    text="Agent note content",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ]

            with patch("main.load_config", return_value=config), patch(
                "main.Retriever.retrieve", return_value=retrieved
            ), patch(
                "main.OllamaChatClient.answer_question",
                return_value="Grounded answer",
            ), patch(
                "main.prompt_to_save"
            ) as prompt_mock, patch(
                "main.save_answer"
            ) as save_mock, patch(
                "sys.argv",
                ["main.py", "ask", "What do my notes say?", "--auto-save"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            prompt_mock.assert_not_called()
            save_mock.assert_called_once()

    def test_main_ask_command_auto_saves_when_config_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            config = replace(config, auto_save_answer=True)
            retrieved = [
                RetrievedChunk(
                    text="Agent note content",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ]

            with patch("main.load_config", return_value=config), patch(
                "main.Retriever.retrieve", return_value=retrieved
            ), patch(
                "main.OllamaChatClient.answer_question",
                return_value="Grounded answer",
            ), patch(
                "main.prompt_to_save"
            ) as prompt_mock, patch(
                "main.save_answer"
            ) as save_mock, patch(
                "sys.argv",
                ["main.py", "ask", "What do my notes say?"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            prompt_mock.assert_not_called()
            save_mock.assert_called_once()
