"""Tests for incremental indexing and retrieval filters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import AppConfig
from main import run_index
from utils import RetrievalFilters
from vector_store import VectorStore


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


class IncrementalIndexingTests(unittest.TestCase):
    def test_index_skips_unchanged_notes_and_updates_changed_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (root / "output").mkdir()
            note_path = vault / "agents.md"
            note_path.write_text("# Agents\n\nInitial content", encoding="utf-8")
            config = make_config(root)

            embed_calls: list[list[str]] = []

            def fake_embed_texts(texts: list[str]) -> list[list[float]]:
                embed_calls.append(list(texts))
                return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embed_texts):
                run_index(config, reset_store=True)
                first_count = VectorStore(config).count()
                run_index(config, reset_store=False)
                second_count = VectorStore(config).count()

                note_path.write_text("# Agents\n\nUpdated content", encoding="utf-8")
                run_index(config, reset_store=False)
                third_count = VectorStore(config).count()

                note_path.unlink()
                run_index(config, reset_store=False)
                fourth_count = VectorStore(config).count()

            self.assertEqual(first_count, 1)
            self.assertEqual(second_count, 1)
            self.assertEqual(third_count, 1)
            self.assertEqual(fourth_count, 0)
            self.assertEqual(len(embed_calls), 2)

    def test_vector_store_applies_folder_and_path_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            store = VectorStore(config)
            store.reset()

            from chunker import chunk_notes
            from utils import Note

            notes = [
                Note(path="projects/agents.md", title="Agents", content="# Agents\n\nAgent systems use tools."),
                Note(path="ideas/notes.md", title="Ideas", content="# Ideas\n\nGeneral brainstorming."),
            ]
            chunks = chunk_notes(notes, chunk_size=1000, overlap=100)
            embeddings = [[1.0, 0.0], [0.0, 1.0]]
            store.upsert_chunks(chunks, embeddings)

            folder_results = store.query([1.0, 0.0], 3, filters=RetrievalFilters(folder="projects"))
            path_results = store.query([1.0, 0.0], 3, filters=RetrievalFilters(path_contains="agents"))

            self.assertEqual(len(folder_results), 1)
            self.assertEqual(folder_results[0].metadata["source_dir"], "projects")
            self.assertEqual(len(path_results), 1)
            self.assertEqual(path_results[0].metadata["source_path"], "projects/agents.md")
