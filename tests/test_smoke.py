"""Smoke tests for local modules."""

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

    def test_load_notes_ignores_configured_output_folder_inside_vault(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault = Path(tmp_dir)
            output_dir = vault / "research_answers"
            output_dir.mkdir()
            (vault / "source.md").write_text("# Source\nkeep", encoding="utf-8")
            (output_dir / "saved_answer.md").write_text("# Saved\nignore", encoding="utf-8")

            notes = load_notes(vault, excluded_paths=[output_dir])

            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].title, "Source")

    def test_load_notes_parses_frontmatter_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault = Path(tmp_dir)
            (vault / "tagged.md").write_text(
                "---\n"
                "category: research\n"
                "tags:\n"
                "  - ai\n"
                "  - agents\n"
                "---\n\n"
                "# Tagged Note\n\n"
                "This note mentions #local-rag in the body.\n",
                encoding="utf-8",
            )

            notes = load_notes(vault)

            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].frontmatter, {"category": "research", "tags": ["ai", "agents"]})
            self.assertEqual(notes[0].tags, ("ai", "agents", "local-rag"))

    def test_load_notes_extracts_obsidian_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault = Path(tmp_dir)
            (vault / "linked.md").write_text(
                "# Linked Note\n\n"
                "See [[Local LLMs]] and [[pkm#Links]].\n",
                encoding="utf-8",
            )

            notes = load_notes(vault)

            self.assertEqual(notes[0].links, ("local llms", "pkm"))


class ChunkerTests(unittest.TestCase):
    def test_chunk_notes_preserves_metadata(self) -> None:
        note = Note(path="ideas/test.md", title="Test", content="A" * 1200)
        chunks = chunk_notes([note], chunk_size=1000, overlap=100)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].note_title, "Test")
        self.assertEqual(chunks[0].source_path, "ideas/test.md")
        self.assertEqual(chunks[1].chunk_index, 1)
        self.assertTrue(chunks[0].note_fingerprint)
        self.assertEqual(chunks[0].source_dir, "ideas")

    def test_chunk_notes_prefers_markdown_sections(self) -> None:
        note = Note(
            path="notes/agents.md",
            title="Agents",
            content=(
                "# Agents\n\n"
                "Intro paragraph.\n\n"
                "## Retrieval\n\n"
                "Retrieval grounded systems use note context.\n\n"
                "## Planning\n\n"
                "Planning helps break down hard tasks."
            ),
        )

        chunks = chunk_notes([note], chunk_size=120, overlap=20)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading_context, "Agents")
        self.assertIn("Retrieval", {chunk.heading_context for chunk in chunks})

    def test_chunk_notes_supports_sentence_strategy(self) -> None:
        note = Note(
            path="notes/agents.md",
            title="Agents",
            content=(
                "# Agents\n\n"
                "Agents plan carefully. Agents use tools effectively. Agents rely on context. "
                "Good retrieval improves grounded answers."
            ),
        )

        chunks = chunk_notes([note], chunk_size=55, overlap=10, strategy="sentence")

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading_context, "Agents")
        self.assertTrue(any("Agents use tools effectively." in chunk.text for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
