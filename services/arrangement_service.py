"""Best-effort parsing helpers for track arrangement markdown notes."""

from __future__ import annotations

import re

from metadata_parser import parse_markdown_metadata
from services.models import (
    ArrangementDocument,
    ArrangementSection,
    ArrangementSectionIndexEntry,
)


TRACK_ARRANGEMENT_TEMPLATE = """---
type: track_arrangement
track_name: Your Track
genre: progressive_house
bpm: 124
key: A minor
status: draft
reference_tracks:
  - Reference Track Name
arrangement_version: 1
---

# Arrangement Overview

## Global Notes
- Overall goal:
- Main concern:

## Section Index
| ID | Name | Bars | Start Bar | End Bar | Energy | Themes |
|---|---|---:|---:|---:|---:|---|
| S1 | Intro | 8 | 1 | 8 | 2 | none |
| S2 | Main Groove | 16 | 9 | 24 | 4 | theme_a |

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

### Transitions / Automation
- keep energy restrained

### Issues / Opportunities
- could use subtle atmos movement
"""


class ArrangementService:
    """Parse arrangement markdown into lightweight structured models."""

    def is_arrangement_document(self, frontmatter: dict[str, object] | None) -> bool:
        return str((frontmatter or {}).get("type", "")).strip().lower() == "track_arrangement"

    def parse_markdown(self, raw_markdown: str) -> ArrangementDocument:
        """Parse raw markdown with frontmatter into an arrangement document."""
        frontmatter, body = parse_markdown_metadata(raw_markdown)
        return self.parse(frontmatter, body)

    def parse(
        self,
        frontmatter: dict[str, object] | None,
        body: str,
    ) -> ArrangementDocument:
        """Parse frontmatter and markdown body into a best-effort arrangement model."""
        frontmatter = frontmatter or {}
        section_index = self._parse_section_index(body)
        sections = self._parse_sections(body, section_index)

        return ArrangementDocument(
            track_id=_clean_text(frontmatter.get("track_id")),
            track_name=_clean_text(frontmatter.get("track_name")),
            total_bars=_calculate_total_bars(section_index, sections),
            genre=_clean_text(frontmatter.get("genre")),
            bpm=_coerce_int(frontmatter.get("bpm")),
            key=_clean_text(frontmatter.get("key")),
            status=_clean_text(frontmatter.get("status")),
            reference_tracks=_as_list(frontmatter.get("reference_tracks")),
            arrangement_version=_coerce_version(frontmatter.get("arrangement_version")),
            global_notes=_extract_bullets(self._extract_heading_block(body, "Global Notes")),
            section_index=section_index,
            sections=sections,
        )

    def render_overview_chunk(self, arrangement: ArrangementDocument) -> str:
        """Render arrangement overview content for arrangement-aware retrieval."""
        lines = ["# Arrangement Overview"]
        summary_lines = [
            f"Track ID: {arrangement.track_id}" if arrangement.track_id else "",
            f"Track: {arrangement.track_name}" if arrangement.track_name else "",
            f"Genre: {arrangement.genre}" if arrangement.genre else "",
            f"BPM: {arrangement.bpm}" if arrangement.bpm is not None else "",
            f"Key: {arrangement.key}" if arrangement.key else "",
            f"Status: {arrangement.status}" if arrangement.status else "",
            f"Total Bars: {arrangement.total_bars}" if arrangement.total_bars is not None else "",
            (
                f"Arrangement Version: {arrangement.arrangement_version}"
                if arrangement.arrangement_version is not None
                else ""
            ),
        ]
        if any(summary_lines):
            lines.extend(["", *[line for line in summary_lines if line]])
        if arrangement.reference_tracks:
            lines.extend(["", "Reference Tracks:", *[f"- {track}" for track in arrangement.reference_tracks]])
        if arrangement.global_notes:
            lines.extend(["", "## Global Notes", *[f"- {note}" for note in arrangement.global_notes]])
        if arrangement.section_index:
            lines.append("")
            lines.append("## Section Index")
            for entry in arrangement.section_index:
                themes = f" | Themes: {', '.join(entry.themes)}" if entry.themes else ""
                bars = f" | Bars: {entry.bars}" if entry.bars else ""
                energy = f" | Energy: {entry.energy}" if entry.energy is not None else ""
                lines.append(f"- {entry.id} | {entry.name}{bars}{energy}{themes}")
        return "\n".join(lines).strip()

    def render_section_chunk(self, arrangement: ArrangementDocument, section: ArrangementSection) -> str:
        """Render a self-contained arrangement section chunk for retrieval."""
        heading = f"# {section.id} - {section.name}".strip()
        lines = [heading]
        context_lines = [
            f"Track: {arrangement.track_name}" if arrangement.track_name else "",
            (
                f"Bars: {section.start_bar}-{section.end_bar}"
                if section.start_bar is not None and section.end_bar is not None
                else ""
            ),
            f"Energy: {section.energy}" if section.energy is not None else "",
            f"Purpose: {section.purpose}" if section.purpose else "",
        ]
        if any(context_lines):
            lines.extend(["", *[line for line in context_lines if line]])
        if section.elements:
            lines.extend(["", "## Key Elements", *[f"- {item}" for item in section.elements]])
        if section.notes:
            lines.extend(["", "## Notes", *[f"- {item}" for item in section.notes]])
        if section.issues:
            lines.extend(["", "## Issues / Opportunities", *[f"- {item}" for item in section.issues]])
        return "\n".join(lines).strip()

    def _parse_section_index(self, body: str) -> list[ArrangementSectionIndexEntry]:
        block = self._extract_heading_block(body, "Section Index")
        rows = _parse_markdown_table(block)
        entries: list[ArrangementSectionIndexEntry] = []
        for row in rows:
            entry_id = _clean_text(row.get("id"))
            name = _clean_text(row.get("name"))
            if not entry_id and not name:
                continue
            bars = _clean_text(row.get("bars"))
            start_bar = _coerce_int(row.get("start bar")) or _parse_bar_range(bars)[0]
            end_bar = _coerce_int(row.get("end bar")) or _parse_bar_range(bars)[1]
            entries.append(
                ArrangementSectionIndexEntry(
                    id=entry_id or name or f"S{len(entries) + 1}",
                    name=name or entry_id or f"Section {len(entries) + 1}",
                    bars=bars or None,
                    start_bar=start_bar,
                    end_bar=end_bar,
                    energy=_coerce_int(row.get("energy")),
                    themes=_split_themes(row.get("themes")),
                )
            )
        return entries

    def _parse_sections(
        self,
        body: str,
        section_index: list[ArrangementSectionIndexEntry],
    ) -> list[ArrangementSection]:
        section_rows = {entry.id.lower(): entry for entry in section_index if entry.id}
        sections: list[ArrangementSection] = []
        for heading, block in _extract_sections(body):
            section_id, section_name = _parse_section_heading(heading, len(sections) + 1)
            lines = [line for line in block.splitlines()]
            fields = _parse_simple_fields(lines)
            index_entry = section_rows.get(section_id.lower())
            bars_text = _first_non_empty(_clean_text(fields.get("bars")), index_entry.bars if index_entry else "")
            start_bar, end_bar = _parse_bar_range(bars_text)
            subsection_map = _extract_subsections(block)
            elements = self._parse_elements(
                subsection_map.get("active layers", ""),
                themes=_split_themes(fields.get("themes")) or (list(index_entry.themes) if index_entry else []),
            )
            notes = _merge_unique_items(
                _extract_bullets(
                    _first_non_empty(
                        subsection_map.get("transitions / automation", ""),
                        subsection_map.get("transitions", ""),
                        subsection_map.get("automation", ""),
                    )
                ),
                _extract_bullets(subsection_map.get("notes", "")),
            )
            sections.append(
                ArrangementSection(
                    id=section_id,
                    name=section_name or (index_entry.name if index_entry else f"Section {len(sections) + 1}"),
                    start_bar=start_bar or (index_entry.start_bar if index_entry else None),
                    end_bar=end_bar or (index_entry.end_bar if index_entry else None),
                    energy=_coerce_int(fields.get("energy")) or (index_entry.energy if index_entry else None),
                    elements=elements,
                    notes=notes,
                    purpose=_clean_text(fields.get("purpose")),
                    issues=_extract_bullets(
                        _first_non_empty(
                            subsection_map.get("issues / opportunities", ""),
                            subsection_map.get("issues", ""),
                            subsection_map.get("opportunities", ""),
                        )
                    ),
                )
            )
        return sections

    def _parse_elements(self, block: str, *, themes: list[str]) -> list[str]:
        rows = _parse_markdown_table(block)
        elements: list[str] = []
        for row in rows:
            layer = _clean_text(row.get("layer"))
            state = _clean_text(row.get("state"))
            if not layer:
                continue
            notes = _clean_text(row.get("notes"))
            normalized_state = (state or "").lower()
            if normalized_state in {"", "on", "active", "full"}:
                label = layer
            elif normalized_state == "off":
                continue
            else:
                label = f"{layer} ({state})"
            if notes:
                label = f"{label} - {notes}"
            elements.append(label)
        for theme in themes:
            if theme.lower() == "none":
                continue
            elements.append(f"Theme: {theme}")
        return _merge_unique_items(elements)

    def _extract_heading_block(self, body: str, heading: str) -> str:
        normalized_heading = heading.strip().lower()
        lines = body.splitlines()
        collected: list[str] = []
        in_block = False
        block_level = 0
        for raw_line in lines:
            stripped = raw_line.strip()
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                level = len(heading_match.group(1))
                current_heading = heading_match.group(2).strip().lower()
                if in_block and level <= block_level:
                    break
                if current_heading == normalized_heading:
                    in_block = True
                    block_level = level
                    continue
            if in_block:
                collected.append(raw_line.rstrip())
        return "\n".join(line for line in collected if line.strip()).strip()


def _extract_sections(body: str) -> list[tuple[str, str]]:
    lines = body.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        match = re.match(r"^##\s+(.*)$", stripped)
        if match:
            heading = match.group(1).strip()
            if re.match(r"^[A-Za-z0-9_]+\s*-\s*.+$", heading):
                if current_heading and current_lines:
                    sections.append((current_heading, "\n".join(current_lines).strip()))
                current_heading = heading
                current_lines = []
                continue
            if current_heading:
                current_lines.append(raw_line.rstrip())
                continue
        if current_heading:
            current_lines.append(raw_line.rstrip())

    if current_heading and current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def _extract_subsections(block: str) -> dict[str, str]:
    subsection_map: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        match = re.match(r"^###\s+(.*)$", stripped)
        if match:
            if current_heading:
                subsection_map[current_heading] = "\n".join(current_lines).strip()
            current_heading = match.group(1).strip().lower()
            current_lines = []
            continue
        if current_heading:
            current_lines.append(raw_line.rstrip())

    if current_heading:
        subsection_map[current_heading] = "\n".join(current_lines).strip()
    return subsection_map


def _parse_simple_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("|") or stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _parse_section_heading(heading: str, fallback_index: int) -> tuple[str, str]:
    match = re.match(r"^(?P<id>[A-Za-z0-9_]+)\s*-\s*(?P<name>.+)$", heading.strip())
    if match:
        return match.group("id").strip(), match.group("name").strip()
    cleaned = heading.strip()
    return f"S{fallback_index}", cleaned or f"Section {fallback_index}"


def _parse_markdown_table(block: str) -> list[dict[str, str]]:
    table_lines = [line.strip() for line in block.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 2:
        return []

    headers = [_normalize_table_cell(cell).lower() for cell in table_lines[0].strip("|").split("|")]
    if not headers or all(not header for header in headers):
        return []

    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        values = [_normalize_table_cell(cell) for cell in line.strip("|").split("|")]
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        rows.append({header: value for header, value in zip(headers, values) if header})
    return rows


def _normalize_table_cell(value: str) -> str:
    return value.strip().strip("`")


def _parse_bar_range(value: str) -> tuple[int | None, int | None]:
    match = re.search(r"(?P<start>\d+)\s*-\s*(?P<end>\d+)", value or "")
    if match:
        return int(match.group("start")), int(match.group("end"))
    single_bar = _coerce_int(value)
    if single_bar is not None:
        return single_bar, single_bar
    return None, None


def _extract_bullets(block: str) -> list[str]:
    items: list[str] = []
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if item:
                items.append(item)
    return items


def _split_themes(value: object) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return []
    lowered = cleaned.lower()
    if lowered == "none":
        return []
    return [part.strip() for part in re.split(r"[;,]", cleaned) if part.strip()]


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    cleaned = _clean_text(value)
    return [cleaned] if cleaned else []


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", str(value).strip())
    return int(match.group(0)) if match else None


def _coerce_version(value: object) -> int | str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    coerced = _coerce_int(cleaned)
    return coerced if coerced is not None and cleaned == str(coerced) else cleaned


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _calculate_total_bars(
    section_index: list[ArrangementSectionIndexEntry],
    sections: list[ArrangementSection],
) -> int | None:
    indexed_end_bars = [entry.end_bar for entry in section_index if entry.end_bar is not None]
    section_end_bars = [section.end_bar for section in sections if section.end_bar is not None]
    all_end_bars = indexed_end_bars + section_end_bars
    return max(all_end_bars) if all_end_bars else None


def _merge_unique_items(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            cleaned = _clean_text(item)
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(cleaned)
    return merged
