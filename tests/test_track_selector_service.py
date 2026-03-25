"""Tests for legacy markdown track selection from Projects/."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.track_selector_service import TrackSelectorService, selected_track_index, selected_track_path


class TrackSelectorServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TrackSelectorService()

    def test_missing_projects_folder_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            vault_path.mkdir()
            self.assertEqual(self.service.list_tracks(vault_path), [])

    def test_finds_valid_track_folders_and_ignores_invalid_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            projects_path = vault_path / "Projects"
            (projects_path / "Moonlit Driver").mkdir(parents=True)
            (projects_path / "Moonlit Driver" / "track_context.md").write_text("", encoding="utf-8")
            (projects_path / "Ideas").mkdir(parents=True)
            tracks = self.service.list_tracks(vault_path)

            self.assertEqual(
                tracks,
                [{"name": "Moonlit Driver", "path": "Projects/Moonlit Driver/track_context.md"}],
            )

    def test_returns_relative_paths_sorted_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            projects_path = vault_path / "Projects"
            (projects_path / "Zulu Track").mkdir(parents=True)
            (projects_path / "Zulu Track" / "track_context.md").write_text("", encoding="utf-8")
            (projects_path / "Alpha Track").mkdir(parents=True)
            (projects_path / "Alpha Track" / "track_context.md").write_text("", encoding="utf-8")

            tracks = self.service.list_tracks(vault_path)

            self.assertEqual(
                tracks,
                [
                    {"name": "Alpha Track", "path": "Projects/Alpha Track/track_context.md"},
                    {"name": "Zulu Track", "path": "Projects/Zulu Track/track_context.md"},
                ],
            )

    def test_finds_nested_project_track_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            nested_track = vault_path / "Projects" / "Current Tracks" / "Moonlit Driver"
            nested_track.mkdir(parents=True)
            (nested_track / "track_context.md").write_text("", encoding="utf-8")

            tracks = self.service.list_tracks(vault_path)

            self.assertEqual(
                tracks,
                [
                    {
                        "name": "Current Tracks / Moonlit Driver",
                        "path": "Projects/Current Tracks/Moonlit Driver/track_context.md",
                    }
                ],
            )

    def test_selected_track_path_and_index_support_ui_wiring(self) -> None:
        tracks = [
            {"name": "Alpha Track", "path": "Projects/Alpha Track/track_context.md"},
            {"name": "Zulu Track", "path": "Projects/Zulu Track/track_context.md"},
        ]

        self.assertEqual(
            selected_track_path("Zulu Track", tracks),
            "Projects/Zulu Track/track_context.md",
        )
        self.assertIsNone(selected_track_path("None", tracks))
        self.assertEqual(
            selected_track_index("Projects/Zulu Track/track_context.md", tracks),
            2,
        )
