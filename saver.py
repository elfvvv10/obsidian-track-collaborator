"""Save generated answers back into the Obsidian vault."""

from __future__ import annotations

from pathlib import Path

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

    return (
        f"# Research Answer\n\n"
        f"**Timestamp:** {current_timestamp()}\n\n"
        f"## Question\n\n"
        f"{question}\n\n"
        f"## Answer\n\n"
        f"{result.answer}\n\n"
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
