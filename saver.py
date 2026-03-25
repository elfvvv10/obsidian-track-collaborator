"""Save generated answers back into the Obsidian vault."""

from __future__ import annotations

from pathlib import Path
import re

from services.models import TrackContext
from utils import AnswerResult, current_timestamp, ensure_directory, slugify


def prompt_to_save() -> bool:
    """Prompt the user to save the answer into the output folder."""
    response = input("\nSave this answer to your Obsidian draft answers folder? (y/n): ").strip().lower()
    return response in {"y", "yes"}


def save_answer(
    output_path: Path,
    question: str,
    result: AnswerResult,
    *,
    title_override: str | None = None,
    source_type: str = "saved_answer",
    status: str = "draft",
    indexed: bool = False,
    domain_profile: str | None = None,
    workflow_type: str | None = None,
    workflow_input: dict[str, str] | None = None,
    track_context: TrackContext | None = None,
) -> Path:
    """Write the answer and sources to a markdown file."""
    ensure_directory(output_path)

    date_prefix = current_timestamp().split(" ")[0]
    title_for_slug = title_override.strip() if title_override and title_override.strip() else question
    file_name = f"{date_prefix}-{slugify(title_for_slug, max_length=40)}-answer.md"
    destination = _unique_destination(output_path / file_name)

    body = _build_markdown(
        question,
        result,
        title_override=title_override,
        source_type=source_type,
        status=status,
        indexed=indexed,
        domain_profile=domain_profile,
        workflow_type=workflow_type,
        workflow_input=workflow_input or {},
        track_context=track_context,
    )
    destination.write_text(body, encoding="utf-8")
    return destination


def _build_markdown(
    question: str,
    result: AnswerResult,
    *,
    title_override: str | None = None,
    source_type: str,
    status: str,
    indexed: bool,
    domain_profile: str | None,
    workflow_type: str | None,
    workflow_input: dict[str, str],
    track_context: TrackContext | None,
) -> str:
    sources = "\n".join(f"- {source}" for source in result.sources) or "- No sources available"
    summary = _build_summary(result.answer)
    key_points = _build_key_points(result.answer)
    key_points_block = "\n".join(f"- {point}" for point in key_points) or "- No key points extracted"
    title = title_override.strip() if title_override and title_override.strip() else "Research Answer"
    timestamp = current_timestamp()
    structured_input_block = _build_structured_input_block(workflow_input)
    track_context_block = format_track_context_summary(track_context)
    actionability_block = _build_actionability_block(workflow_type, result.answer)
    inference_note = (
        "This output includes explicitly labeled inference or stylistic extrapolation."
        if "[Inference]" in result.answer
        else "No explicit inference labels were used in this output."
    )
    workflow_section_title = _workflow_section_title(workflow_type)
    frontmatter_lines = [
        "---",
        f'source_type: "{_escape_frontmatter(source_type)}"',
        f'status: "{_escape_frontmatter(status)}"',
        f"indexed: {'true' if indexed else 'false'}",
        'created_by: "obsidian_rag_assistant"',
        f'created_at: "{timestamp}"',
        f'original_question: "{_escape_frontmatter(question)}"',
        f'saved_at: "{timestamp}"',
    ]
    if domain_profile:
        frontmatter_lines.append(f'domain_profile: "{_escape_frontmatter(domain_profile)}"')
    if workflow_type:
        frontmatter_lines.append(f'workflow_type: "{_escape_frontmatter(workflow_type)}"')
    if workflow_input:
        frontmatter_lines.append("workflow_input:")
        for key, value in workflow_input.items():
            frontmatter_lines.append(f'  {key}: "{_escape_frontmatter(value)}"')
    frontmatter = "\n".join(frontmatter_lines) + "\n---"

    return (
        f"{frontmatter}\n\n"
        f"# {title}\n\n"
        f"**Timestamp:** {timestamp}\n\n"
        f"**Workflow:** {workflow_type or 'general_ask'}\n\n"
        f"## Question\n\n"
        f"{question}\n\n"
        f"## Input Summary\n\n"
        f"{structured_input_block}\n\n"
        f"{track_context_block}"
        f"## Summary\n\n"
        f"{summary}\n\n"
        f"## {workflow_section_title}\n\n"
        f"{result.answer}\n\n"
        f"## Key Points\n\n"
        f"{key_points_block}\n\n"
        f"## Inference Notes\n\n"
        f"{inference_note}\n\n"
        f"{actionability_block}\n\n"
        f"## Sources\n\n"
        f"{sources}\n"
    )


def _unique_destination(path: Path) -> Path:
    """Avoid overwriting an existing saved answer note."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _build_summary(answer: str) -> str:
    text = " ".join(line.strip() for line in answer.splitlines() if line.strip())
    if not text:
        return "No summary available."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = sentences[0].strip()
    return summary or text[:160].strip()


def _build_key_points(answer: str) -> list[str]:
    bullet_points: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            bullet_points.append(stripped[2:].strip())

    if bullet_points:
        return bullet_points[:5]

    text = " ".join(line.strip() for line in answer.splitlines() if line.strip())
    if not text:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return sentences[:3]


def _escape_frontmatter(value: str) -> str:
    return value.replace('"', '\\"')


def _build_structured_input_block(workflow_input: dict[str, str]) -> str:
    if not workflow_input:
        return "- No additional structured workflow fields were provided."
    return "\n".join(
        f"- {key.replace('_', ' ').title()}: {value}"
        for key, value in workflow_input.items()
    )


def format_track_context_summary(track_context: TrackContext | None) -> str:
    """Render a concise markdown summary for YAML-backed track context."""
    if track_context is None:
        return ""

    lines = [
        "## Track Context",
        "",
        f"- Track ID: {track_context.track_id}",
        f"- Workflow Mode: {track_context.workflow_mode}",
    ]
    optional_fields = (
        ("Track Name", track_context.track_name),
        ("Genre", track_context.genre),
        ("BPM", track_context.bpm),
        ("Key", track_context.key),
        ("Current Stage", track_context.current_stage),
        ("Current Section", track_context.current_section),
    )
    for label, value in optional_fields:
        if value is not None and str(value).strip():
            lines.append(f"- {label}: {value}")
    if track_context.vibe:
        lines.append(f"- Vibe: {', '.join(track_context.vibe)}")
    if track_context.reference_tracks:
        lines.append(f"- Reference Tracks: {', '.join(track_context.reference_tracks)}")
    if track_context.known_issues:
        lines.append(f"- Known Issues: {', '.join(track_context.known_issues)}")
    if track_context.goals:
        lines.append(f"- Goals: {', '.join(track_context.goals)}")
    if track_context.notes:
        lines.append(f"- Notes: {' | '.join(track_context.notes)}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def _workflow_section_title(workflow_type: str | None) -> str:
    return {
        "genre_fit_review": "Genre Fit Review",
        "track_concept_critique": "Track Concept Critique",
        "arrangement_planner": "Arrangement Plan",
        "sound_design_brainstorm": "Sound Design Brainstorm",
        "research_session": "Research Session",
    }.get(workflow_type or "", "Answer")


def _build_actionability_block(workflow_type: str | None, answer: str) -> str:
    section_title = {
        "genre_fit_review": "## Style Alignment Notes",
        "track_concept_critique": "## Suggested Next Steps",
        "arrangement_planner": "## Production Plan Notes",
        "sound_design_brainstorm": "## Practical Production Notes",
        "research_session": "## Research Follow-Up",
    }.get(workflow_type or "", "## Action Notes")
    points = _build_key_points(answer)
    point_lines = "\n".join(f"- {point}" for point in points) or "- No action notes extracted."
    return f"{section_title}\n\n{point_lines}"
