"""Tests for Phase 1 retrieval improvements."""

from __future__ import annotations

import unittest

from llm import build_prompt
from reranker import rerank_chunks
from utils import RetrievedChunk


class RerankerTests(unittest.TestCase):
    def test_reranker_promotes_lexically_relevant_chunk(self) -> None:
        chunks = [
            RetrievedChunk(
                text="General note about planning.",
                metadata={"note_title": "Planning", "source_path": "planning.md"},
                distance_or_score=0.05,
            ),
            RetrievedChunk(
                text="AI agents use tools and retrieval for grounded answers.",
                metadata={"note_title": "Agents", "source_path": "agents.md"},
                distance_or_score=0.25,
            ),
        ]

        reranked = rerank_chunks("How do AI agents use retrieval tools?", chunks)

        self.assertEqual(reranked[0].metadata["note_title"], "Agents")

    def test_reranker_can_promote_imported_note_title_match(self) -> None:
        chunks = [
            RetrievedChunk(
                text="This arrangement uses low-energy intro layers and restrained transitions.",
                metadata={
                    "note_title": "Minimal Arrangement Reference",
                    "source_path": "Knowledge/Arrangement/minimal.md",
                    "content_category": "curated_knowledge",
                },
                distance_or_score=0.05,
            ),
            RetrievedChunk(
                text="Producer workflow and mixing details.",
                metadata={
                    "note_title": "Boris Brejcha Tutorial Breakdown",
                    "source_path": "Imports/YouTube Imports/Generic/boris.md",
                    "content_category": "imported_knowledge",
                    "source_kind": "imported_content",
                },
                distance_or_score=0.2,
            ),
        ]

        reranked = rerank_chunks("boris brejcha", chunks)

        self.assertEqual(reranked[0].metadata["note_title"], "Boris Brejcha Tutorial Breakdown")


class PromptFormattingTests(unittest.TestCase):
    def test_build_prompt_includes_structured_context(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Agents use retrieval to ground answers.",
                metadata={
                    "note_title": "Agents",
                    "source_path": "notes/agents.md",
                    "heading_context": "Retrieval",
                },
                distance_or_score=0.12345,
            )
        ]

        prompt = build_prompt("What do my notes say about agents?", chunks)

        self.assertIn("[Source 1]", prompt)
        self.assertIn("Title: Agents | Section: Retrieval", prompt)
        self.assertIn("Path: notes/agents.md", prompt)
        self.assertIn("Relevance distance: 0.1235", prompt)
        self.assertIn("Content:", prompt)
        self.assertIn("Tags: ai, agents", build_prompt("Q", [RetrievedChunk(
            text="Tagged chunk",
            metadata={
                "note_title": "Agents",
                "source_path": "notes/agents.md",
                "heading_context": "Retrieval",
                "tags_serialized": "ai|agents",
            },
            distance_or_score=0.1,
        )]))
