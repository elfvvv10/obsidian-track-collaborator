# Vault Organization

This project works with a wide range of Obsidian vault layouts, but testing and day-to-day use are much easier when the vault follows a clear structure.

The recommendations below are intentionally conservative:
- existing folders still work
- legacy markdown `track_context.md` files still work
- current save paths still work
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

Research/
  genre_style/
  production_techniques/
  references/
  imported_material/

Saved Outputs/
  answers/
  research/
  critiques/

Templates/
  track_context_template.md
  session_note_template.md

Archive/
```

## Folder Purpose

### `Projects/`
- active track and project work
- one folder per track
- the best home for `track_context.md`
- use subfolders like `session_notes/`, `arrangements/`, and `sound_design/` for focused working material

### `Research/`
- general reference material that is not tied to one active track
- genre breakdowns, production notes, references, imports, and broader study material

### `Saved Outputs/`
- app-generated outputs you want to keep easy to browse
- direct answers, research outputs, and critiques can live here if you update your config accordingly

### `Templates/`
- reusable starting points for new tracks and sessions

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

Current app behavior remains compatible with existing `Drafts/` and `Research Sessions/` paths.

If you want a cleaner vault, you can gradually move toward:
- `Saved Outputs/answers/`
- `Saved Outputs/research/`
- `Saved Outputs/critiques/`

This project does not force that change automatically.

## Legacy Compatibility

Older layouts still behave as before, including:
- `Drafts/...`
- `Research Sessions/...`
- direct track folders under `Projects/...`
- nested project folders such as `Projects/Current Tracks/...`

The app now discovers legacy markdown `track_context.md` files recursively under `Projects/` to support both older and cleaner layouts.

## Suggested Adoption Path

1. Keep your existing vault as-is.
2. Create the recommended top-level folders.
3. Put active tracks under `Projects/Current Tracks/`.
4. Put reusable research under `Research/`.
5. Start new tracks from the templates in `Templates/`.
6. Optionally update config later if you want app-saved outputs to move under `Saved Outputs/`.
