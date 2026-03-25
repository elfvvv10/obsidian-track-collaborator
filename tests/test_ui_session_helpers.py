"""Tests for pure session UI helper formatting."""

from __future__ import annotations

import unittest

from services.models import TrackContext, TrackContextSuggestions
from services.ui_session_helpers import current_track_summary, debug_query_summary, suggestion_groups


class UISessionHelpersTests(unittest.TestCase):
    def test_current_track_summary_handles_missing_context(self) -> None:
        title, rows = current_track_summary(None)
        self.assertEqual(title, "No active YAML Track Context")
        self.assertEqual(rows, [])

    def test_current_track_summary_returns_compact_rows(self) -> None:
        title, rows = current_track_summary(
            TrackContext(
                track_id="moonlit_driver",
                track_name="Moonlit Driver",
                genre="progressive house",
                bpm=126,
                key="A minor",
                workflow_mode="track_critique",
                current_stage="arrangement",
                current_section="first drop",
            )
        )

        self.assertEqual(title, "Active YAML Track Context")
        self.assertIn(("Track ID", "moonlit_driver"), rows)
        self.assertIn(("Workflow Mode", "track_critique"), rows)
        self.assertIn(("Current Section", "first drop"), rows)

    def test_suggestion_groups_returns_clear_grouping(self) -> None:
        groups = suggestion_groups(
            TrackContextSuggestions(
                known_issues=["drop lacks contrast"],
                goals=["increase build tension"],
                current_stage="arrangement",
            )
        )

        self.assertEqual(
            groups,
            [
                ("Known Issues", ["drop lacks contrast"]),
                ("Goals", ["increase build tension"]),
                ("Current Stage", "arrangement"),
            ],
        )

    def test_debug_query_summary_separates_original_and_rewritten_query(self) -> None:
        rows = debug_query_summary(
            "help with the bassline",
            "help with the bassline progressive house first drop",
        )
        self.assertEqual(
            rows,
            [
                ("Original question", "help with the bassline"),
                ("Rewritten retrieval query", "help with the bassline progressive house first drop"),
            ],
        )
