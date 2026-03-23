"""Helpers for parsing and normalizing Obsidian note links."""

from __future__ import annotations

import re

from utils import normalize_path


def extract_obsidian_links(content: str) -> tuple[str, ...]:
    """Extract normalized Obsidian-style note links from markdown content."""
    matches = re.findall(r"\[\[([^\]]+)\]\]", content)
    normalized_links: list[str] = []
    seen: set[str] = set()

    for match in matches:
        normalized = normalize_link_target(match)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_links.append(normalized)

    return tuple(normalized_links)


def normalize_link_target(value: str) -> str:
    """Normalize an Obsidian link target for note resolution."""
    target = value.split("|", 1)[0].split("#", 1)[0].split("^", 1)[0].strip()
    if not target:
        return ""
    if target.lower().endswith(".md"):
        target = target[:-3]
    return normalize_path(target).lower()
