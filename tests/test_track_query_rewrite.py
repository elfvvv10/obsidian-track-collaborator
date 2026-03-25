"""Tests for retrieval-only Track Context query rewriting."""

from __future__ import annotations

import unittest

from services.models import TrackContext
from services.track_query_rewrite_service import TrackQueryRewriteService


class TrackQueryRewriteServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TrackQueryRewriteService()

    def test_no_context_returns_original_question(self) -> None:
        self.assertEqual(self.service.rewrite("help with the bassline", None), "help with the bassline")

    def test_includes_relevant_fields_in_priority_order(self) -> None:
        rewritten = self.service.rewrite(
            "help with the bassline",
            TrackContext(
                genre="progressive house",
                current_stage="production",
                current_problem="drop feels flat",
                known_issues=["drop feels flat"],
                goals=["stronger groove"],
                vibe=["emotional"],
                bpm=126,
                key="A minor",
            ),
        )

        self.assertEqual(
            rewritten,
            "help with the bassline progressive house production drop feels flat stronger groove emotional 126 A minor",
        )

    def test_skips_empty_values(self) -> None:
        rewritten = self.service.rewrite(
            "bassline help",
            TrackContext(
                genre="",
                current_stage=None,
                current_problem="",
                known_issues=[],
                goals=[],
                vibe=[],
                bpm=None,
                key=None,
            ),
        )

        self.assertEqual(rewritten, "bassline help")

    def test_avoids_duplication(self) -> None:
        rewritten = self.service.rewrite(
            "progressive house bassline",
            TrackContext(
                genre="progressive house",
                goals=["progressive house bassline"],
                vibe=["progressive house"],
            ),
        )

        self.assertEqual(rewritten, "progressive house bassline")

    def test_remains_compact(self) -> None:
        rewritten = self.service.rewrite(
            "help with the bassline",
            TrackContext(
                genre="progressive house",
                known_issues=["drop feels flat", "bass lacks movement", "midrange is crowded"],
                goals=["stronger groove", "clearer contrast", "bigger hook"],
                vibe=["emotional", "rolling", "driving"],
            ),
        )

        self.assertNotIn("midrange is crowded", rewritten)
        self.assertNotIn("bigger hook", rewritten)
        self.assertNotIn("driving", rewritten)
