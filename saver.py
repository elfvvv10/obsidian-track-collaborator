"""Save generated answers back into the Obsidian vault."""

from __future__ import annotations

from pathlib import Path
import re

from utils import AnswerResult, current_timestamp, ensure_directory, slugify


def prompt_to_save() -> bool:
    """Prompt the user to save the answer into the output folder."""
    response = input("\nSave this answer to your Obsidian output folder? (y/n): ").strip().lower()
    return response in {"y", "yes"}


def save_answer(output_path: Path, question: str, result: AnswerResult) -> Path:
    """Write the answer and sources to a markdown file."""
    ensure_directory(output_path)

    date_prefix = current_timestamp().split(" ")[0]
    file_name = f"{date_prefix}-{slugify(question, max_length=40)}-answer.md"
    destination = _unique_destination(output_path / file_name)

    body = _build_markdown(question, result)
    destination.write_text(body, encoding="utf-8")
    return destination


def _build_markdown(question: str, result: AnswerResult) -> str:
    sources = "\n".join(f"- {source}" for source in result.sources) or "- No sources available"
    summary = _build_summary(result.answer)
    key_points = _build_key_points(result.answer)
    key_points_block = "\n".join(f"- {point}" for point in key_points) or "- No key points extracted"

    return (
        f"# Research Answer\n\n"
        f"**Timestamp:** {current_timestamp()}\n\n"
        f"## Question\n\n"
        f"{question}\n\n"
        f"## Summary\n\n"
        f"{summary}\n\n"
        f"## Answer\n\n"
        f"{result.answer}\n\n"
        f"## Key Points\n\n"
        f"{key_points_block}\n\n"
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
