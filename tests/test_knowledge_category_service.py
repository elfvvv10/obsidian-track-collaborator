"""Tests for Knowledge-folder category discovery and normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.knowledge_category_service import (
    GENERIC_KNOWLEDGE_CATEGORY_LABEL,
    KnowledgeCategoryService,
)


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="deepseek",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
        curated_knowledge_folder="Knowledge",
    )


class KnowledgeCategoryServiceTests(unittest.TestCase):
    def test_available_categories_discovers_first_level_knowledge_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            (config.curated_knowledge_path / "Arrangement").mkdir(parents=True)
            (config.curated_knowledge_path / "Sound Design").mkdir(parents=True)

            service = KnowledgeCategoryService(config)

            self.assertEqual(service.available_categories(), ["Arrangement", "Sound Design"])

    def test_display_options_includes_generic_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            (config.curated_knowledge_path / "Mixing").mkdir(parents=True)

            options = KnowledgeCategoryService(config).display_options()

            self.assertEqual(options[0], GENERIC_KNOWLEDGE_CATEGORY_LABEL)
            self.assertIn("Mixing", options)

    def test_available_categories_reflects_new_folder_on_next_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.curated_knowledge_path.mkdir(parents=True)
            service = KnowledgeCategoryService(config)

            self.assertEqual(service.available_categories(), [])

            (config.curated_knowledge_path / "References").mkdir()

            self.assertEqual(service.available_categories(), ["References"])

    def test_canonicalize_is_case_insensitive_and_blank_maps_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            (config.curated_knowledge_path / "Arrangement").mkdir(parents=True)
            service = KnowledgeCategoryService(config)

            self.assertEqual(service.canonicalize("arrangement"), "Arrangement")
            self.assertIsNone(service.canonicalize(""))
            self.assertIsNone(service.canonicalize(GENERIC_KNOWLEDGE_CATEGORY_LABEL))

    def test_validate_or_raise_rejects_unknown_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            (config.curated_knowledge_path / "Arrangement").mkdir(parents=True)
            service = KnowledgeCategoryService(config)

            with self.assertRaisesRegex(ValueError, "Available Knowledge categories: Arrangement"):
                service.validate_or_raise("Mixing")
