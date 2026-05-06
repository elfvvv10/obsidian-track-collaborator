"""Tests for path-safe per-track persistence filename helpers."""

from __future__ import annotations

import unittest

from services.track_path_utils import legacy_flat_track_file_stem, safe_track_file_stem


class TrackPathUtilsTests(unittest.TestCase):
    def test_safe_track_file_stem_preserves_simple_existing_ids(self) -> None:
        self.assertEqual(safe_track_file_stem("moonlit_driver"), "moonlit_driver")

    def test_safe_track_file_stem_sanitizes_and_hashes_free_text(self) -> None:
        stem = safe_track_file_stem("../Warehouse Hypnosis/../../bad tune")

        self.assertRegex(stem, r"^Warehouse_Hypnosis_bad_tune_[0-9a-f]{8}$")

    def test_safe_track_file_stem_avoids_sanitized_name_collisions(self) -> None:
        self.assertNotEqual(safe_track_file_stem("a/b"), safe_track_file_stem("a b"))

    def test_legacy_flat_track_file_stem_rejects_traversal(self) -> None:
        self.assertIsNone(legacy_flat_track_file_stem("../bad"))
        self.assertIsNone(legacy_flat_track_file_stem("nested/bad"))
        self.assertIsNone(legacy_flat_track_file_stem(r"nested\bad"))
        self.assertIsNone(legacy_flat_track_file_stem(".."))
        self.assertIsNone(legacy_flat_track_file_stem("."))

    def test_legacy_flat_track_file_stem_rejects_control_characters(self) -> None:
        self.assertIsNone(legacy_flat_track_file_stem("bad\0name"))
        self.assertIsNone(legacy_flat_track_file_stem("bad\nname"))
        self.assertIsNone(legacy_flat_track_file_stem("bad\tname"))

    def test_legacy_flat_track_file_stem_allows_old_flat_friendly_names(self) -> None:
        self.assertEqual(legacy_flat_track_file_stem("Warehouse Hypnosis"), "Warehouse Hypnosis")


if __name__ == "__main__":
    unittest.main()
