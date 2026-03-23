"""Small smoke tests for local modules."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chunker import chunk_notes
from utils import Note
from vault_loader import load_notes


class VaultLoaderTests(unittest.TestCase):
    def test_load_notes_ignores_hidden_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault = Path(tmp_dir)
            (vault / ".obsidian").mkdir()
            (vault / ".obsidian" / "ignored.md").write_text("# Hidden\nignore", encoding="utf-8")
            (vault / "visible.md").write_text("# Visible\nkeep", encoding="utf-8")

            notes = load_notes(vault)

            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].title, "Visible")


class ChunkerTests(unittest.TestCase):
    def test_chunk_notes_preserves_metadata(self) -> None:
        note = Note(path="ideas/test.md", title="Test", content="A" * 1200)
        chunks = chunk_notes([note], chunk_size=1000, overlap=100)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].note_title, "Test")
        self.assertEqual(chunks[0].source_path, "ideas/test.md")
        self.assertEqual(chunks[1].chunk_index, 1)


if __name__ == "__main__":
    unittest.main()
