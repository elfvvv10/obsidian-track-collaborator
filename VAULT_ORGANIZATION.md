# Vault Organization

This project works with a wide range of Obsidian vault layouts, but testing and day-to-day use are much easier when the vault follows a clear structure.

The recommendations below are intentionally conservative:
- existing folders still work
- legacy markdown `track_context.md` files still work
- older save paths can still be indexed if you keep them around
- you can adopt the structure gradually

## Recommended Structure

```text
Projects/
  Current Tracks/
    <Track Name>/
      track_context.md
      session_notes/
      arrangements/
      sound_design/
      exports/
  Track Ideas/
  Archived Tracks/

Knowledge/
  Arrangement/
  Drums and Groove/
  Genres/
  Mixing/
  References/
  Sound Design/

Imports/
  Web Imports/
  YouTube Imports/

Saved Outputs/
  answers/
    General Asks/
    Arrangement Plans/
    Sound Design Brainstorms/
  research/
  critiques/
    Genre Fit Reviews/
    Track Concept Critiques/

Templates/
  track_context_template.md
  session_note_template.md

Sources/
  Frameworks/
    Music Production/
      track_critique_framework_v1.md

Archive/
```

## Folder Purpose

### `Projects/`
- active track and project work
- one folder per track
- the best home for `track_context.md`
- use subfolders like `session_notes/`, `arrangements/`, and `sound_design/` for focused working material

### `Knowledge/`
- your reusable production library
- genre breakdowns, arrangement ideas, mixing notes, and evergreen references

### `Imports/`
- raw imported material such as web and YouTube ingestions
- keep these separate from curated notes until you decide they belong in `Knowledge/`

### `Saved Outputs/`
- the primary home for app-generated outputs
- `answers/` holds direct asks, arrangement plans, and sound-design brainstorms
- `critiques/` holds workflow outputs that read more like reviews
- `research/` holds saved research runs

### `Templates/`
- reusable starting points for new tracks and sessions

### `Sources/`
- project-supporting source material and internal reference documents
- the preferred home for framework notes such as `Sources/Frameworks/Music Production/track_critique_framework_v1.md`

### `Archive/`
- older material you want to keep out of the active working path without deleting

## Track Context Guidance

For track-specific work, prefer:

```text
Projects/Current Tracks/<Track Name>/track_context.md
```

This location is:
- easy for humans to find
- easy for the markdown selector to discover
- consistent with session notes and other track-specific files

The YAML Track Context system remains separate from the legacy markdown path flow. Both can coexist.

## Saved Output Guidance

The app now defaults to:
- `Saved Outputs/answers/`
- `Saved Outputs/research/`
- `Saved Outputs/critiques/`

## Legacy Compatibility

Older layouts still behave as before, including:
- direct track folders under `Projects/...`
- nested project folders such as `Projects/Current Tracks/...`
- older output notes already saved under `Drafts/...`
- older research notes already saved under `Research Sessions/...`

The app now discovers legacy markdown `track_context.md` files recursively under `Projects/` to support both older and cleaner layouts.

If you already have a `Research/` folder, treat it as a legacy library location and gradually fold curated notes into `Knowledge/`.

## Suggested Adoption Path

1. Put active tracks under `Projects/Current Tracks/`.
2. Put reusable notes under `Knowledge/`.
3. Keep raw imports under `Imports/`.
4. Start new tracks from the templates in `Templates/`.
5. Let the app save new outputs under `Saved Outputs/`.
6. Archive or migrate older `Drafts/`, `Research Sessions/`, and `Research/` content as needed.
