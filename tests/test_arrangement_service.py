"""Tests for arrangement parsing and section-aware chunking."""

from __future__ import annotations

import unittest

from chunker import chunk_notes
from services.arrangement_service import ArrangementService
from utils import Note


ARRANGEMENT_MARKDOWN = """---
type: track_arrangement
track_name: My Track
genre: progressive_house
bpm: 124
key: A minor
status: draft
reference_tracks:
  - Tripchain
arrangement_version: 1
---

# Arrangement Overview

## Global Notes
- Overall goal: hypnotic progressive flow with stronger second drop
- Main concern: middle feels too flat before break

## Section Index
| ID | Name | Bars | Start Bar | End Bar | Energy | Themes |
|---|---|---:|---:|---:|---:|---|
| S1 | Intro | 8 | 1 | 8 | 2 | none |
| S2 | Intro 2 | 16 | 9 | 24 | 3 | A hint |

# Sections

## S1 - Intro
Bars: 1-8
Energy: 2
Purpose: establish groove, low commitment, DJ-friendly intro

### Active Layers
| Layer | State | Notes |
|---|---|---|
| kick | partial | no groove hat |
| bass_main | off | |
| sub | off | |
| theme_a | off | |
| impact | sparse | crash only |

### Transitions / Automation
- no groove hat
- keep energy restrained

### Issues / Opportunities
- could use subtle foley or atmos movement

## S2 - Intro 2
Bars: 9-24
Energy: 3
Purpose: add motion and theme hint
Themes: A hint

### Transitions / Automation
- open the hats slightly
"""


class ArrangementServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ArrangementService()

    def test_valid_arrangement_markdown_parses_into_document(self) -> None:
        arrangement = self.service.parse_markdown(ARRANGEMENT_MARKDOWN)

        self.assertEqual(arrangement.track_name, "My Track")
        self.assertEqual(arrangement.genre, "progressive_house")
        self.assertEqual(arrangement.bpm, 124)
        self.assertEqual(arrangement.arrangement_version, 1)
        self.assertEqual(arrangement.global_notes[0], "Overall goal: hypnotic progressive flow with stronger second drop")
        self.assertEqual(len(arrangement.section_index), 2)
        self.assertEqual(len(arrangement.sections), 2)

    def test_section_extraction_and_active_layers_work(self) -> None:
        arrangement = self.service.parse_markdown(ARRANGEMENT_MARKDOWN)
        intro = arrangement.sections[0]

        self.assertEqual(intro.id, "S1")
        self.assertEqual(intro.name, "Intro")
        self.assertEqual(intro.start_bar, 1)
        self.assertEqual(intro.end_bar, 8)
        self.assertEqual(intro.energy, 2)
        self.assertEqual(intro.purpose, "establish groove, low commitment, DJ-friendly intro")
        self.assertEqual(intro.active_layers[0].layer, "kick")
        self.assertEqual(intro.active_layers[0].state, "partial")
        self.assertEqual(intro.active_layers[0].notes, "no groove hat")

    def test_transitions_and_issues_are_captured(self) -> None:
        arrangement = self.service.parse_markdown(ARRANGEMENT_MARKDOWN)
        intro = arrangement.sections[0]

        self.assertEqual(intro.transitions, ["no groove hat", "keep energy restrained"])
        self.assertEqual(intro.issues, ["could use subtle foley or atmos movement"])

    def test_incomplete_arrangement_docs_do_not_crash(self) -> None:
        arrangement = self.service.parse(
            {"type": "track_arrangement", "track_name": "Sketch"},
            "# Sections\n\n## S1 - Intro\nBars: 1-8\n",
        )

        self.assertEqual(arrangement.track_name, "Sketch")
        self.assertEqual(len(arrangement.sections), 1)
        self.assertEqual(arrangement.sections[0].start_bar, 1)
        self.assertEqual(arrangement.sections[0].end_bar, 8)


class ArrangementChunkingTests(unittest.TestCase):
    def test_arrangement_sections_remain_meaningfully_separated(self) -> None:
        note = Note(
            path="Track Context/My Track/arrangement.md",
            title="My Track Arrangement",
            content=ARRANGEMENT_MARKDOWN.split("---\n", 2)[2].strip(),
            frontmatter={
                "type": "track_arrangement",
                "track_name": "My Track",
                "genre": "progressive_house",
                "bpm": 124,
                "arrangement_version": 1,
            },
            source_type="track_arrangement",
        )

        chunks = chunk_notes([note], chunk_size=500, overlap=50)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].arrangement_section_id, "overview")
        self.assertTrue(any(chunk.arrangement_section_id == "S1" for chunk in chunks))
        self.assertTrue(any(chunk.arrangement_section_id == "S2" for chunk in chunks))
        self.assertTrue(any("Active Layers" in chunk.text for chunk in chunks if chunk.arrangement_section_id == "S1"))

    def test_arrangement_metadata_is_preserved_on_chunks(self) -> None:
        note = Note(
            path="Track Context/My Track/arrangement.md",
            title="My Track Arrangement",
            content=ARRANGEMENT_MARKDOWN.split("---\n", 2)[2].strip(),
            frontmatter={
                "type": "track_arrangement",
                "track_name": "My Track",
                "genre": "progressive_house",
                "bpm": 124,
                "arrangement_version": 1,
            },
            source_type="track_arrangement",
        )

        chunks = chunk_notes([note], chunk_size=500, overlap=50)
        intro_chunk = next(chunk for chunk in chunks if chunk.arrangement_section_id == "S1")

        self.assertEqual(intro_chunk.source_type, "track_arrangement")
        self.assertEqual(intro_chunk.arrangement_track_name, "My Track")
        self.assertEqual(intro_chunk.arrangement_genre, "progressive_house")
        self.assertEqual(intro_chunk.arrangement_section_name, "Intro")
        self.assertEqual(intro_chunk.arrangement_energy, 2)
        self.assertEqual(intro_chunk.arrangement_version, "1")
