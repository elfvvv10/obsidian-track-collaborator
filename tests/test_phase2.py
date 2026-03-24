"""Tests for Phase 2 metadata and tag-aware retrieval."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from config import AppConfig
from metadata_parser import extract_tags, parse_markdown_metadata
from retriever import Retriever
from utils import RetrievedChunk, RetrievalFilters, RetrievalOptions
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


class MetadataParserTests(unittest.TestCase):
    def test_parse_markdown_metadata_splits_frontmatter_from_body(self) -> None:
        frontmatter, body = parse_markdown_metadata(
            "---\n"
            "tags: [ai, agents]\n"
            "category: research\n"
            "---\n\n"
            "# Note\n\n"
            "Body text.\n"
        )

        self.assertEqual(frontmatter, {"tags": ["ai", "agents"], "category": "research"})
        self.assertEqual(body, "# Note\n\nBody text.")

    def test_extract_tags_combines_frontmatter_and_inline_tags(self) -> None:
        tags = extract_tags({"tags": ["ai", "agents"]}, "Body mentions #local-rag and #agents.")
        self.assertEqual(tags, ("ai", "agents", "local-rag"))


class TagAwareRetrievalTests(unittest.TestCase):
    def test_vector_store_filters_by_tag(self) -> None:
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
                Note(path="agents.md", title="Agents", content="# Agents\n\nUses tools.", tags=("ai", "agents")),
                Note(path="pkm.md", title="PKM", content="# PKM\n\nPersonal knowledge.", tags=("pkm",)),
            ]
            chunks = chunk_notes(notes, chunk_size=1000, overlap=100)
            store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

            results = store.query([1.0, 0.0], 3, filters=RetrievalFilters(tag="agents"))

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata["note_title"], "Agents")
            self.assertEqual(results[0].metadata["tags_serialized"], "ai|agents")

    def test_retriever_boosts_matching_tags(self) -> None:
        class StubEmbeddingClient:
            def embed_text(self, text: str) -> list[float]:
                return [1.0, 0.0]

        class StubVectorStore:
            def count(self) -> int:
                return 2

            def query(self, query_embedding: list[float], top_k: int, filters=None) -> list[RetrievedChunk]:
                return [
                    RetrievedChunk(
                        "generic note about planning",
                        {"note_title": "Planning", "source_path": "planning.md", "tags_serialized": "planning"},
                        0.05,
                    ),
                    RetrievedChunk(
                        "note about local models",
                        {"note_title": "Local AI", "source_path": "local.md", "tags_serialized": "ai|local"},
                        0.2,
                    ),
                ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())

            results = retriever.retrieve(
                "planning",
                options=RetrievalOptions(top_k=1, candidate_count=2, boost_tags=("ai",)),
            )

            self.assertEqual(results[0].metadata["note_title"], "Local AI")


class Phase2CLITests(unittest.TestCase):
    def test_main_ask_command_passes_tag_filters_and_boosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retrieved = [
                RetrievedChunk(
                    text="Agent note content",
                    metadata={"note_title": "Agents", "source_path": "agents.md", "tags_serialized": "ai|agents"},
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
                ["main.py", "ask", "What do my notes say?", "--tag", "agents", "--boost-tag", "ai"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            called_filters = retrieve_mock.call_args.kwargs["filters"]
            called_options = retrieve_mock.call_args.kwargs["options"]
            self.assertEqual(called_filters.tag, "agents")
            self.assertEqual(called_options.boost_tags, ("ai",))
