"""Tests for pure session UI helper formatting."""

from __future__ import annotations

import unittest

from services.models import (
    CollaborationWorkflow,
    TrackContext,
    TrackContextSuggestions,
    TrackContextUpdateProposal,
)
from services.track_context_update_review import proposal_groups
from services.ui_session_helpers import (
    DEV_MODE_PRESET_FAST,
    DEV_MODE_PRESET_LOCAL,
    DEV_MODE_PRESET_MANUAL,
    DEV_MODE_PRESET_QUALITY,
    critique_support_summary,
    current_track_summary,
    dev_mode_preset_options,
    debug_query_summary,
    resolve_dev_mode_preset,
    synced_dev_mode_preset_selection,
    synced_chat_provider_selection,
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
        self.assertIn(("Title", "Moonlit Driver"), rows)
        self.assertIn(("References", "Tripchain"), rows)
        self.assertIn(("Current Problem", "drop lacks contrast"), rows)

    def test_track_context_status_describes_loaded_existing_context(self) -> None:
        title, caption = track_context_status(
            use_track_context=True,
            entered_track_id="moonlit_driver",
            active_track_id="moonlit_driver",
            existed_before_load=True,
            track_context=TrackContext(track_id="moonlit_driver", track_name="Moonlit Driver"),
        )

        self.assertIn("Loaded existing track memory", title)
        self.assertIn("active in-progress track", caption)

    def test_track_context_status_describes_ready_to_load_track(self) -> None:
        title, caption = track_context_status(
            use_track_context=True,
            entered_track_id="warehouse-hypnosis-01",
            active_track_id="",
            existed_before_load=False,
            track_context=None,
        )

        self.assertIn("ready to load", title)
        self.assertIn("Load Track Context", caption)

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

    def test_suggestion_groups_includes_extended_track_context_fields(self) -> None:
        groups = suggestion_groups(
            TrackContextSuggestions(
                vibe_suggestions=["dark", "driving"],
                reference_track_suggestions=["Bicep - Glue"],
                section_suggestions={"drop": {"issues": ["lacks contrast"], "elements": ["riser"]}},
                section_focus="drop",
                bpm_suggestion=126,
                key_suggestion="F#m",
            )
        )

        self.assertEqual(
            groups,
            [
                ("Vibe", ["dark", "driving"]),
                ("Reference Tracks", ["Bicep - Glue"]),
                ("BPM", "126"),
                ("Key", "F#m"),
                ("Section Focus", "drop"),
                ("Section: Drop", ["Issues: lacks contrast", "Elements: riser"]),
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

    def test_proposal_groups_formats_update_sections_compactly(self) -> None:
        groups = proposal_groups(
            TrackContextUpdateProposal(
                summary="Capture the current drop issue.",
                set_fields={"genre": "progressive house"},
                add_to_lists={"next_actions": ["shorten the fill before the drop"]},
                set_sections={"drop": {"role": "main payoff", "energy_level": "high"}},
                section_focus="drop",
            )
        )

        self.assertEqual(
            groups,
            [
                ("Set Fields", ["genre: progressive house"]),
                ("Add To Lists", ["next_actions: shorten the fill before the drop"]),
                ("Set Sections", ["drop: role=main payoff | energy_level=high"]),
                ("Section Focus", ["drop"]),
            ],
        )

    def test_synced_chat_provider_selection_initializes_from_configured_default(self) -> None:
        selection, synced = synced_chat_provider_selection(
            current_selection="",
            committed_override="",
            configured_provider="openai",
            last_synced_override="",
        )

        self.assertEqual(selection, "Use configured default (openai)")
        self.assertEqual(synced, "")

    def test_synced_chat_provider_selection_preserves_pending_user_edit(self) -> None:
        selection, synced = synced_chat_provider_selection(
            current_selection="ollama",
            committed_override="",
            configured_provider="openai",
            last_synced_override="",
        )

        self.assertEqual(selection, "ollama")
        self.assertEqual(synced, "")

    def test_synced_chat_provider_selection_resyncs_when_committed_override_changes(self) -> None:
        selection, synced = synced_chat_provider_selection(
            current_selection="openai",
            committed_override="ollama",
            configured_provider="openai",
            last_synced_override="",
        )

        self.assertEqual(selection, "ollama")
        self.assertEqual(synced, "ollama")

    def test_dev_mode_preset_options_include_required_presets(self) -> None:
        self.assertEqual(
            dev_mode_preset_options(),
            [
                DEV_MODE_PRESET_MANUAL,
                DEV_MODE_PRESET_FAST,
                DEV_MODE_PRESET_QUALITY,
                DEV_MODE_PRESET_LOCAL,
            ],
        )

    def test_synced_dev_mode_preset_selection_preserves_pending_user_edit(self) -> None:
        selection, synced = synced_dev_mode_preset_selection(
            current_selection=DEV_MODE_PRESET_LOCAL,
            committed_preset="",
            last_synced_preset="",
        )

        self.assertEqual(selection, DEV_MODE_PRESET_LOCAL)
        self.assertEqual(synced, "")

    def test_synced_dev_mode_preset_selection_resyncs_to_committed_preset(self) -> None:
        selection, synced = synced_dev_mode_preset_selection(
            current_selection=DEV_MODE_PRESET_MANUAL,
            committed_preset=DEV_MODE_PRESET_FAST,
            last_synced_preset="",
        )

        self.assertEqual(selection, DEV_MODE_PRESET_FAST)
        self.assertEqual(synced, DEV_MODE_PRESET_FAST)

    def test_resolve_dev_mode_preset_returns_openai_fast_settings(self) -> None:
        self.assertEqual(
            resolve_dev_mode_preset(
                DEV_MODE_PRESET_FAST,
                configured_ollama_model="deepseek-r1:latest",
                available_ollama_models=["deepseek-r1:latest"],
            ),
            ("openai", "gpt-4.1-mini"),
        )

    def test_resolve_dev_mode_preset_prefers_configured_ollama_model(self) -> None:
        self.assertEqual(
            resolve_dev_mode_preset(
                DEV_MODE_PRESET_LOCAL,
                configured_ollama_model="deepseek-r1:latest",
                available_ollama_models=["llama3.2", "deepseek-r1:latest"],
            ),
            ("ollama", "deepseek-r1:latest"),
        )
