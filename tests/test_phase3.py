"""Tests for Phase 3 note-link awareness."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from config import AppConfig
from link_parser import extract_obsidian_links, normalize_link_target
from retriever import Retriever
from utils import Note, RetrievedChunk, RetrievalOptions


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


class LinkParserTests(unittest.TestCase):
    def test_extract_obsidian_links_normalizes_variants(self) -> None:
        links = extract_obsidian_links(
            "See [[Local LLMs]] and [[notes/ai_agents.md|AI Agents]] plus [[PKM#Links]]."
        )
        self.assertEqual(links, ("local llms", "notes/ai_agents", "pkm"))

    def test_normalize_link_target_strips_alias_heading_and_extension(self) -> None:
        self.assertEqual(normalize_link_target("folder/Note.md|Alias"), "folder/note")
        self.assertEqual(normalize_link_target("PKM#Links"), "pkm")


class LinkResolutionTests(unittest.TestCase):
    def test_resolve_note_links_maps_links_to_note_keys(self) -> None:
        notes = [
            Note(path="ai_agents.md", title="AI Agents", content="...", links=("local llms",)),
            Note(path="local_llms.md", title="Local LLMs", content="..."),
        ]

        main._resolve_note_links(notes)

        self.assertEqual(len(notes[0].linked_note_keys), 1)
        self.assertEqual(notes[1].linked_note_keys, ())


class LinkedContextRetrievalTests(unittest.TestCase):
    def test_retriever_appends_linked_note_chunks_when_enabled(self) -> None:
        class StubEmbeddingClient:
            def embed_text(self, text: str) -> list[float]:
                return [1.0, 0.0]

        class StubVectorStore:
            def count(self) -> int:
                return 2

            def query(self, query_embedding: list[float], top_k: int, filters=None) -> list[RetrievedChunk]:
                return [
                    RetrievedChunk(
                        "primary chunk",
                        {
                            "note_title": "Agents",
                            "source_path": "ai_agents.md",
                            "note_key": "note-1",
                            "linked_note_keys_serialized": "note-2",
                        },
                        0.1,
                    )
                ]

            def get_chunks_by_note_keys(self, note_keys, *, max_chunks_per_note, excluded_note_keys=None):
                self.request = (note_keys, max_chunks_per_note, excluded_note_keys)
                return [
                    RetrievedChunk(
                        "linked chunk",
                        {
                            "note_title": "Local LLMs",
                            "source_path": "local_llms.md",
                            "linked_context": True,
                        },
                        None,
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())

            results = retriever.retrieve(
                "question",
                options=RetrievalOptions(top_k=1, candidate_count=1, include_linked_notes=True),
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(results[1].metadata["note_title"], "Local LLMs")
            self.assertTrue(results[1].metadata["linked_context"])


class Phase3CLITests(unittest.TestCase):
    def test_main_ask_command_passes_include_linked_option(self) -> None:
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
            ) as retrieve_mock, patch(
                "main.OllamaChatClient.answer_with_prompt",
                return_value="Grounded answer",
            ), patch(
                "main.prompt_to_save",
                return_value=False,
            ), patch(
                "sys.argv",
                ["main.py", "ask", "What do my notes say?", "--include-linked"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            called_options = retrieve_mock.call_args.kwargs["options"]
            self.assertTrue(called_options.include_linked_notes)
