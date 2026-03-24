"""Answer-mode prompt policies and citation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from services.models import AnswerMode, RetrievalMode
from utils import RetrievedChunk
from web_search import WebSearchResult


@dataclass(slots=True)
class PromptPayload:
    """Structured prompt payload for the chat layer."""

    system_prompt: str
    user_prompt: str
    answer_mode: AnswerMode
    citation_labels: tuple[str, ...]
    evidence_types_used: tuple[str, ...]


class PromptService:
    """Build answer-mode-specific prompts while keeping policy out of the UI."""

    def build_prompt_payload(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        web_results: list[WebSearchResult],
        retrieval_mode: RetrievalMode,
        answer_mode: AnswerMode,
        local_retrieval_weak: bool,
        web_query_used: str = "",
        web_query_strategy: str = "raw_question",
        web_alignment_note: str = "",
    ) -> PromptPayload:
        citation_sources, citation_labels = build_citation_sources(chunks, web_results)
        evidence_types_used = _build_evidence_types(chunks, web_results)
        return PromptPayload(
            system_prompt=_build_system_prompt(answer_mode),
            user_prompt=_build_user_prompt(
                question,
                chunks,
                web_results=web_results,
                retrieval_mode=retrieval_mode,
                answer_mode=answer_mode,
                local_retrieval_weak=local_retrieval_weak,
                citation_sources=citation_sources,
                web_query_used=web_query_used,
                web_query_strategy=web_query_strategy,
                web_alignment_note=web_alignment_note,
            ),
            answer_mode=answer_mode,
            citation_labels=tuple(citation_labels),
            evidence_types_used=evidence_types_used,
        )


def build_citation_sources(
    chunks: list[RetrievedChunk],
    web_results: list[WebSearchResult],
) -> tuple[list[str], list[str]]:
    """Build stable, labeled source strings for answers and UI display."""
    labeled_sources: list[str] = []
    citation_labels: list[str] = []
    seen_sources: set[str] = set()

    local_index = 1
    for chunk in chunks:
        key = (
            chunk.metadata.get("note_title", "Untitled"),
            chunk.metadata.get("source_path", "unknown"),
        )
        if key in seen_sources:
            continue
        seen_sources.add(key)
        label = f"[Local {local_index}]"
        labeled_sources.append(
            f"{label} {chunk.metadata.get('note_title', 'Untitled')} "
            f"({chunk.metadata.get('source_path', 'unknown')})"
        )
        citation_labels.append(label)
        local_index += 1

    web_index = 1
    for result in web_results:
        key = (result.title, result.url)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        label = f"[Web {web_index}]"
        labeled_sources.append(f"{label} {result.title} ({result.url})")
        citation_labels.append(label)
        web_index += 1

    return labeled_sources, citation_labels


def enforce_citation_summary(answer: str, citation_labels: tuple[str, ...], answer_mode: AnswerMode) -> str:
    """Append a short citation summary when evidence exists but labels are missing."""
    if not citation_labels:
        return answer
    if any(label in answer for label in citation_labels):
        return answer

    if answer_mode == AnswerMode.STRICT:
        suffix = "Citations: " + ", ".join(citation_labels)
    elif answer_mode == AnswerMode.BALANCED:
        suffix = "Evidence used: " + ", ".join(citation_labels)
    else:
        suffix = "Evidence anchors: " + ", ".join(citation_labels)
    return f"{answer.rstrip()}\n\n{suffix}"


def answer_uses_inference(answer: str) -> bool:
    """Detect whether the model explicitly labeled inference in its answer."""
    return "[Inference]" in answer


def _build_system_prompt(answer_mode: AnswerMode) -> str:
    common = (
        "You are a careful research assistant for an Obsidian vault. "
        "Keep local-note evidence and external web evidence distinct. "
        "Use citation labels exactly as provided when referencing evidence."
    )
    if answer_mode == AnswerMode.STRICT:
        return (
            common
            + " Strict mode: use only retrieved evidence, refuse when evidence is insufficient, "
            "and do not present unsupported claims as facts."
        )
    if answer_mode == AnswerMode.EXPLORATORY:
        return (
            common
            + " Exploratory mode: you may synthesize and infer beyond direct evidence, but label "
            "those parts with [Inference] and keep supported claims cited."
        )
    return (
        common
        + " Balanced mode: prioritize retrieved evidence, allow limited synthesis, and label any "
        "beyond-evidence reasoning with [Inference]."
    )


def _build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    web_results: list[WebSearchResult],
    retrieval_mode: RetrievalMode,
    answer_mode: AnswerMode,
    local_retrieval_weak: bool,
    citation_sources: list[str],
    web_query_used: str,
    web_query_strategy: str,
    web_alignment_note: str,
) -> str:
    local_context_block = _format_local_context(chunks)
    web_context_block = _format_web_context(web_results)
    citation_block = "\n".join(citation_sources) if citation_sources else "No evidence sources were retrieved."
    evidence_strength = "weak" if local_retrieval_weak else "usable"
    mode_instructions = _mode_instructions(answer_mode)

    return (
        f"Answer mode: {answer_mode.value}\n"
        f"Retrieval mode: {retrieval_mode.value}\n"
        f"Local retrieval strength: {evidence_strength}\n\n"
        f"Web query used: {web_query_used or question}\n"
        f"Web query strategy: {web_query_strategy}\n"
        f"Web alignment note: {web_alignment_note or 'No special web alignment was applied.'}\n\n"
        "Citation labels:\n"
        f"{citation_block}\n\n"
        "Local note context:\n"
        f"{local_context_block}\n\n"
        "External web evidence:\n"
        f"{web_context_block}\n\n"
        f"Question: {question}\n\n"
        f"{mode_instructions}\n"
    )


def _mode_instructions(answer_mode: AnswerMode) -> str:
    if answer_mode == AnswerMode.STRICT:
        return (
            "Strict mode instructions:\n"
            "- Use only the provided evidence.\n"
            "- If the evidence is missing, weak, or insufficient, say so directly.\n"
            "- Cite supported statements with [Local N] or [Web N].\n"
            "- Do not use unstated model knowledge as factual support."
        )
    if answer_mode == AnswerMode.EXPLORATORY:
        return (
            "Exploratory mode instructions:\n"
            "- Use the provided evidence first.\n"
            "- You may add broader synthesis or extrapolation, but prefix those parts with [Inference].\n"
            "- Cite supported claims with [Local N] or [Web N].\n"
            "- Make it clear when a point goes beyond direct evidence.\n"
            "- Only use web evidence when it is directly relevant to the local-note topic."
        )
    return (
        "Balanced mode instructions:\n"
        "- Prioritize the provided evidence.\n"
        "- You may connect supported ideas with limited reasoning.\n"
        "- Label any beyond-evidence synthesis with [Inference].\n"
        "- Cite supported claims with [Local N] or [Web N].\n"
        "- If web evidence is broader than the local topic, say that explicitly instead of blending it in."
    )


def _format_local_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant local note context was retrieved."
    parts: list[str] = []
    local_index = 1
    for chunk in chunks:
        title = chunk.metadata.get("note_title", "Untitled note")
        source_path = chunk.metadata.get("source_path", "unknown")
        heading_context = chunk.metadata.get("heading_context", "")
        section_line = f" | Section: {heading_context}" if heading_context else ""
        context_kind = "Linked note" if chunk.metadata.get("linked_context") else "Primary retrieval"
        label = f"[Local {local_index}]"
        local_index += 1
        parts.append(
            f"{label}\n"
            f"Type: {context_kind}\n"
            f"Title: {title}{section_line}\n"
            f"Path: {source_path}\n"
            f"Content:\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _format_web_context(web_results: list[WebSearchResult]) -> str:
    if not web_results:
        return "No external web evidence was used."
    parts: list[str] = []
    for index, result in enumerate(web_results, start=1):
        label = f"[Web {index}]"
        parts.append(
            f"{label}\n"
            f"Title: {result.title}\n"
            f"URL: {result.url}\n"
            f"Snippet:\n{result.snippet}"
        )
    return "\n\n".join(parts)


def _build_evidence_types(
    chunks: list[RetrievedChunk],
    web_results: list[WebSearchResult],
) -> tuple[str, ...]:
    evidence_types: list[str] = []
    if chunks:
        evidence_types.append("local_note")
    if web_results:
        evidence_types.append("web")
    return tuple(evidence_types)
