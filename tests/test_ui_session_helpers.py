"""Tests for pure session UI helper formatting."""

from __future__ import annotations

import unittest

from services.models import CollaborationWorkflow, TrackContext, TrackContextSuggestions
from services.ui_session_helpers import (
    critique_support_summary,
    current_track_summary,
    debug_query_summary,
    suggestion_groups,
    track_context_status,
)
from utils import RetrievedChunk


class UISessionHelpersTests(unittest.TestCase):
    def test_current_track_summary_handles_missing_context(self) -> None:
        title, caption, rows = current_track_summary(None, use_track_context=False)
        self.assertEqual(title, "Current Track")
        self.assertEqual(caption, "No active YAML Track Context")
        self.assertEqual(rows, [])

    def test_current_track_summary_returns_compact_rows(self) -> None:
        title, caption, rows = current_track_summary(
            TrackContext(
                track_id="moonlit_driver",
                track_name="Moonlit Driver",
                genre="progressive house",
                bpm=126,
                key="A minor",
                vibe=["driving", "euphoric"],
                reference_tracks=["Tripchain"],
                current_stage="arrangement",
                current_problem="drop lacks contrast",
            )
        )

        self.assertEqual(title, "Current Track")
        self.assertIn("Moonlit Driver", caption)
        self.assertIn(("Track ID", "moonlit_driver"), rows)
        self.assertIn(("Vibe", "driving, euphoric"), rows)
        self.assertIn(("Reference Tracks", "Tripchain"), rows)
        self.assertIn(("Current Problem", "drop lacks contrast"), rows)

    def test_track_context_status_describes_loaded_existing_context(self) -> None:
        title, caption = track_context_status(
            use_track_context=True,
            track_id="moonlit_driver",
            existed_before_load=True,
            track_context=TrackContext(track_id="moonlit_driver", track_name="Moonlit Driver"),
        )

        self.assertIn("Loaded existing track memory", title)
        self.assertIn("saved YAML Track Context", caption)

    def test_critique_support_summary_distinguishes_arrangement_support(self) -> None:
        title, lines = critique_support_summary(
            CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            TrackContext(track_id="moonlit_driver", track_name="Moonlit Driver"),
            [
                RetrievedChunk(
                    text="Bars: 33-48\nEnergy: 7",
                    metadata={"source_type": "track_arrangement"},
                )
            ],
        ) or ("", [])

        self.assertEqual(title, "Track-aware critique with arrangement support")
        self.assertTrue(any("Arrangement notes were retrieved" in line for line in lines))

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
