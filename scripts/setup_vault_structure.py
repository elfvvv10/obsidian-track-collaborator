"""Create a recommended non-destructive Obsidian vault folder structure."""

from __future__ import annotations

import argparse
from pathlib import Path


RECOMMENDED_DIRECTORIES = (
    "Projects/Current Tracks",
    "Projects/Track Ideas",
    "Projects/Archived Tracks",
    "Knowledge/Arrangement",
    "Knowledge/Drums and Groove",
    "Knowledge/Genres",
    "Knowledge/Mixing",
    "Knowledge/References",
    "Knowledge/Sound Design",
    "Imports/Web Imports",
    "Imports/YouTube Imports",
    "Sources/Frameworks/Music Production",
    "Saved Outputs/answers/General Asks",
    "Saved Outputs/answers/Arrangement Plans",
    "Saved Outputs/answers/Sound Design Brainstorms",
    "Saved Outputs/research",
    "Saved Outputs/critiques/Genre Fit Reviews",
    "Saved Outputs/critiques/Track Concept Critiques",
    "Templates",
    "Archive",
)


TRACK_CONTEXT_TEMPLATE = """---
track_id: your_track_id
track_name: Your Track Name
genre: ""
bpm:
key: ""
workflow_mode: general
current_stage:
current_section:
vibe: []
reference_tracks: []
sections: {}
known_issues: []
goals: []
notes: []
---
"""


SESSION_NOTE_TEMPLATE = """# Session Note Template

## Track

- Track:
- Date:
- Focus:
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a recommended Obsidian vault structure.")
    parser.add_argument("vault_path", help="Path to the Obsidian vault")
    args = parser.parse_args()

    vault_path = Path(args.vault_path).expanduser().resolve()
    vault_path.mkdir(parents=True, exist_ok=True)

    for relative_path in RECOMMENDED_DIRECTORIES:
        (vault_path / relative_path).mkdir(parents=True, exist_ok=True)

    _write_if_missing(vault_path / "Templates" / "track_context_template.md", TRACK_CONTEXT_TEMPLATE)
    _write_if_missing(vault_path / "Templates" / "session_note_template.md", SESSION_NOTE_TEMPLATE)

    print(f"Recommended vault structure ensured at: {vault_path}")


def _write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
