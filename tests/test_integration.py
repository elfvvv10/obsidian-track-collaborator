"""Higher-value integration tests for orchestration and retrieval behavior."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent import ResearchAgent
from config import AppConfig
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from main import run_index
from retriever import Retriever
from services.models import RetrievalScope
from utils import RetrievalFilters, RetrievalOptions, RetrievedChunk
from vector_store import VectorStore


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="hermes3",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=2,
    )


def fake_embedding_for_text(text: str) -> list[float]:
    lower = text.lower()
    if "garden" in lower:
        return [0.0, 1.0]
    if "local llm" in lower or "ollama" in lower:
        return [0.8, 0.2]
    return [1.0, 0.0]


def fake_embedding_for_texts(texts: list[str]) -> list[list[float]]:
    return [fake_embedding_for_text(text) for text in texts]


class StubChatClient:
    def answer_question(self, question: str, chunks: list[RetrievedChunk]) -> str:
        titles = ", ".join(str(chunk.metadata.get("note_title")) for chunk in chunks)
        return f"Used: {titles}"

    def answer_with_prompt(self, prompt_payload) -> str:
        titles = ", ".join(prompt_payload.citation_labels)
        return f"Used: {titles}"


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_index_and_answer_flow_uses_expected_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (vault / "knowledge").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            (vault / "knowledge" / "ai_agents.md").write_text(
                "---\n"
                "tags: [ai, agents]\n"
                "---\n\n"
                "# AI Agents\n\n"
                "AI agents use retrieval and tools to ground answers.\n",
                encoding="utf-8",
            )
            (vault / "gardening.md").write_text(
                "# Gardening\n\n"
                "Tomatoes need sun and water.\n",
                encoding="utf-8",
            )

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embedding_for_texts):
                run_index(config, reset_store=True)

            retriever = Retriever(config, _StubEmbeddingClient(), VectorStore(config))
            agent = ResearchAgent(retriever, StubChatClient())
            result = agent.answer(
                "What do my notes say about AI agents?",
                options=RetrievalOptions(top_k=1, candidate_count=2),
                retrieval_scope=RetrievalScope.KNOWLEDGE,
            )

            self.assertIn("AI Agents", result.answer)
            self.assertIn("AI Agents (knowledge/ai_agents.md)", result.sources)
            self.assertNotIn("Gardening (gardening.md)", result.sources)

    def test_real_tag_filtering_changes_retrieved_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (vault / "knowledge").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            (vault / "knowledge" / "ai_agents.md").write_text(
                "---\n"
                "tags: [agents]\n"
                "---\n\n"
                "# AI Agents\n\n"
                "AI agents use tools.\n",
                encoding="utf-8",
            )
            (vault / "pkm.md").write_text(
                "---\n"
                "tags: [pkm]\n"
                "---\n\n"
                "# PKM\n\n"
                "Personal knowledge management note.\n",
                encoding="utf-8",
            )

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embedding_for_texts):
                run_index(config, reset_store=True)

            retriever = Retriever(config, _StubEmbeddingClient(), VectorStore(config))
            filtered = retriever.retrieve(
                "What do my notes say about agents?",
                filters=RetrievalFilters(tag="agents"),
                options=RetrievalOptions(top_k=2, candidate_count=2),
                retrieval_scope=RetrievalScope.KNOWLEDGE,
            )

            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].metadata["note_title"], "AI Agents")

    def test_real_linked_note_expansion_includes_linked_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (vault / "knowledge").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            (vault / "knowledge" / "ai_agents.md").write_text(
                "# AI Agents\n\n"
                "AI agents use tools. See [[Local LLMs]].\n",
                encoding="utf-8",
            )
            (vault / "knowledge" / "local_llms.md").write_text(
                "# Local LLMs\n\n"
                "Local LLMs are often run with Ollama. This connects back to [[AI Agents]].\n",
                encoding="utf-8",
            )

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embedding_for_texts):
                run_index(config, reset_store=True)

            retriever = Retriever(config, _StubEmbeddingClient(), VectorStore(config))
            results = retriever.retrieve(
                "What do my notes say about AI agents?",
                options=RetrievalOptions(top_k=1, candidate_count=1, include_linked_notes=True),
                retrieval_scope=RetrievalScope.KNOWLEDGE,
            )

            self.assertGreaterEqual(len(results), 2)
            self.assertIn(results[0].metadata["note_title"], {"AI Agents", "Local LLMs"})
            self.assertTrue(any(chunk.metadata.get("linked_context") for chunk in results[1:]))
            self.assertGreaterEqual(len({chunk.metadata.get("note_title") for chunk in results}), 2)

    def test_knowledge_scope_excludes_non_curated_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (vault / "knowledge").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            (vault / "knowledge" / "agents.md").write_text("# Agents\n\nCurated agent note.", encoding="utf-8")
            (vault / "scratch.md").write_text("# Scratch\n\nNon-curated agent note.", encoding="utf-8")

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embedding_for_texts):
                run_index(config, reset_store=True)

            retriever = Retriever(config, _StubEmbeddingClient(), VectorStore(config))
            results = retriever.retrieve(
                "What do my notes say about AI agents?",
                options=RetrievalOptions(top_k=5, candidate_count=5),
                retrieval_scope=RetrievalScope.KNOWLEDGE,
            )

            self.assertTrue(results)
            self.assertTrue(all(chunk.metadata.get("content_scope") == "knowledge" for chunk in results))

    def test_knowledge_scope_includes_imported_content_when_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (vault / "knowledge").mkdir()
            (vault / "ingested_youtube").mkdir()
            (root / "output").mkdir()
            config = replace(make_config(root), index_youtube_imports=True)

            (vault / "knowledge" / "agents.md").write_text("# Agents\n\nCurated agent note.", encoding="utf-8")
            (vault / "ingested_youtube" / "set.md").write_text(
                '---\nsource_type: "youtube_import"\n---\n\n# DJ Set Breakdown\n\nImported production note.',
                encoding="utf-8",
            )

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embedding_for_texts):
                run_index(config, reset_store=True)

            retriever = Retriever(config, _StubEmbeddingClient(), VectorStore(config))
            results = retriever.retrieve(
                "What knowledge do my notes contain?",
                options=RetrievalOptions(top_k=5, candidate_count=5),
                retrieval_scope=RetrievalScope.KNOWLEDGE,
            )

            categories = {chunk.metadata.get("content_category") for chunk in results}
            self.assertIn("curated_knowledge", categories)
            self.assertIn("imported_knowledge", categories)
            self.assertNotIn("non_curated_note", categories)
            self.assertTrue(all("scratch.md" not in str(chunk.metadata.get("source_path")) for chunk in results))

    def test_extended_scope_can_include_non_curated_and_imported_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (vault / "knowledge").mkdir()
            (vault / "ingested_webpages").mkdir()
            (root / "output").mkdir()
            config = replace(make_config(root), index_webpage_imports=True)

            (vault / "knowledge" / "agents.md").write_text("# Agents\n\nCurated agent note.", encoding="utf-8")
            (vault / "scratch.md").write_text("# Scratch\n\nNon-curated agent note.", encoding="utf-8")
            (vault / "ingested_webpages" / "page.md").write_text(
                '---\nsource_type: "webpage_import"\n---\n\n# Imported Page\n\nImported agent note.',
                encoding="utf-8",
            )

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embedding_for_texts):
                run_index(config, reset_store=True)

            retriever = Retriever(config, _StubEmbeddingClient(), VectorStore(config))
            results = retriever.retrieve(
                "What do my notes say about AI agents?",
                options=RetrievalOptions(top_k=5, candidate_count=5),
                retrieval_scope=RetrievalScope.EXTENDED,
            )

            categories = {chunk.metadata.get("content_category") for chunk in results}
            self.assertIn("curated_knowledge", categories)
            self.assertIn("non_curated_note", categories)
            self.assertIn("imported_knowledge", categories)


class _StubEmbeddingClient(OllamaEmbeddingClient):
    def __init__(self) -> None:
        pass

    def embed_text(self, text: str) -> list[float]:
        return fake_embedding_for_text(text)
