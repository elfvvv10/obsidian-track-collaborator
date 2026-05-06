"""Tests for environment loading and config behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import load_config


class ConfigLoadingTests(unittest.TestCase):
    def test_load_config_reads_values_from_dotenv_in_current_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / ".env").write_text(
                "\n".join(
                    [
                        "OBSIDIAN_VAULT_PATH=./vault",
                        "OBSIDIAN_OUTPUT_PATH=./output",
                        "CHROMA_DB_PATH=./chroma",
                        "TOP_K_RESULTS=7",
                        "WEB_SEARCH_PROVIDER=duckduckgo",
                        "CHAT_PROVIDER=ollama",
                        "EMBEDDING_PROVIDER=ollama",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {}, clear=True):
                    config = load_config()
            finally:
                os.chdir(original_cwd)

            self.assertEqual(config.obsidian_vault_path, (root / "vault").resolve())
            self.assertEqual(config.obsidian_output_path, (root / "output").resolve())
            self.assertEqual(config.chroma_db_path, (root / "chroma").resolve())
            self.assertEqual(config.top_k_results, 7)
            self.assertEqual(config.web_search_provider, "duckduckgo")

    def test_explicit_environment_variables_override_dotenv_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / ".env").write_text(
                "\n".join(
                    [
                        "OBSIDIAN_VAULT_PATH=./vault",
                        "OBSIDIAN_OUTPUT_PATH=./output",
                        "CHROMA_DB_PATH=./chroma",
                        "CHAT_PROVIDER=ollama",
                        "EMBEDDING_PROVIDER=ollama",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(
                    os.environ,
                    {
                        "CHAT_PROVIDER": "openai",
                        "OPENAI_API_KEY": "test-key",
                        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
                    },
                    clear=True,
                ):
                    config = load_config()
            finally:
                os.chdir(original_cwd)

            self.assertEqual(config.chat_provider, "openai")
            self.assertEqual(config.openai_api_key, "test-key")
            self.assertEqual(config.openai_chat_model, "gpt-4o-mini")

    def test_load_config_works_without_dotenv_when_process_env_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {
                "OBSIDIAN_VAULT_PATH": str(Path(tmp_dir) / "vault"),
                "OBSIDIAN_OUTPUT_PATH": str(Path(tmp_dir) / "output"),
                "CHROMA_DB_PATH": str(Path(tmp_dir) / "chroma"),
                "CHAT_PROVIDER": "ollama",
                "EMBEDDING_PROVIDER": "ollama",
            },
            clear=True,
        ):
            (Path(tmp_dir) / "vault").mkdir()
            original_cwd = Path.cwd()
            try:
                os.chdir(tmp_dir)
                config = load_config()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(config.chat_provider, "ollama")
        self.assertEqual(config.openai_api_key, "")
        self.assertEqual(config.openai_chat_model, "")

    def test_load_config_reads_track_critique_framework_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {
                "OBSIDIAN_VAULT_PATH": str(Path(tmp_dir) / "vault"),
                "OBSIDIAN_OUTPUT_PATH": str(Path(tmp_dir) / "output"),
                "CHROMA_DB_PATH": str(Path(tmp_dir) / "chroma"),
                "CHAT_PROVIDER": "ollama",
                "EMBEDDING_PROVIDER": "ollama",
                "TRACK_CRITIQUE_FRAMEWORK_VERSION": "v2",
            },
            clear=True,
        ):
            (Path(tmp_dir) / "vault").mkdir()
            original_cwd = Path.cwd()
            try:
                os.chdir(tmp_dir)
                config = load_config()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(config.track_critique_framework_version, "v2")
