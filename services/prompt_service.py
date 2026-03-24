"""Answer-mode prompt policies and citation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from config import AppConfig
from services.framework_service import FrameworkService
from services.models import (
    AnswerMode,
    CollaborationWorkflow,
    DomainProfile,
    RetrievalMode,
    WorkflowInput,
)
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
    domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC
    collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.GENERAL_ASK


class PromptService:
    """Build answer-mode-specific prompts while keeping policy out of the UI."""

    def __init__(
        self,
        config: AppConfig,
        *,
        framework_service_cls: type[FrameworkService] = FrameworkService,
    ) -> None:
        self.config = config
        self.framework_service = framework_service_cls(config)

    def build_prompt_payload(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        web_results: list[WebSearchResult],
        retrieval_mode: RetrievalMode,
        answer_mode: AnswerMode,
        local_retrieval_weak: bool,
        domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC,
        collaboration_workflow: CollaborationWorkflow = CollaborationWorkflow.GENERAL_ASK,
        workflow_input: WorkflowInput | None = None,
        web_query_used: str = "",
        web_query_strategy: str = "raw_question",
        web_alignment_note: str = "",
    ) -> PromptPayload:
        citation_sources, citation_labels = build_citation_sources(chunks, web_results)
        evidence_types_used = _build_evidence_types(chunks, web_results)
        workflow_input = workflow_input or WorkflowInput()
        framework_text = self.framework_service.get_framework_text(collaboration_workflow, domain_profile)
        return PromptPayload(
            system_prompt=_build_system_prompt(
                answer_mode,
                domain_profile=domain_profile,
                collaboration_workflow=collaboration_workflow,
                framework_text=framework_text,
            ),
            user_prompt=_build_user_prompt(
                question,
                chunks,
                web_results=web_results,
                retrieval_mode=retrieval_mode,
                answer_mode=answer_mode,
                local_retrieval_weak=local_retrieval_weak,
                domain_profile=domain_profile,
                collaboration_workflow=collaboration_workflow,
                workflow_input=workflow_input,
                citation_sources=citation_sources,
                web_query_used=web_query_used,
                web_query_strategy=web_query_strategy,
                web_alignment_note=web_alignment_note,
            ),
            answer_mode=answer_mode,
            citation_labels=tuple(citation_labels),
            evidence_types_used=evidence_types_used,
            domain_profile=domain_profile,
            collaboration_workflow=collaboration_workflow,
        )

    def build_research_plan_payload(
        self,
        goal: str,
        *,
        answer_mode: AnswerMode,
        max_subquestions: int,
        domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC,
        workflow_input: WorkflowInput | None = None,
    ) -> PromptPayload:
        """Build a lightweight planning prompt for subquestion generation."""
        workflow_input = workflow_input or WorkflowInput()
        return PromptPayload(
            system_prompt=(
                "You are planning an explicit research workflow for an Obsidian-based electronic music research assistant. "
                "Break the user's goal into focused, evidence-seeking subquestions. "
                "Be concrete, avoid overlap, and do not answer the question yet."
            ),
            user_prompt=(
                f"Goal: {goal}\n"
                f"Domain profile: {domain_profile.value}\n"
                f"Answer mode: {answer_mode.value}\n"
                f"Structured workflow context:\n{_format_workflow_input(workflow_input)}\n"
                f"Generate {max_subquestions} or fewer focused subquestions.\n"
                "Keep the subquestions useful for electronic music production, critique, arrangement, sound design, or style research when relevant.\n"
                "Return one subquestion per line with no numbering or extra commentary."
            ),
            answer_mode=answer_mode,
            citation_labels=(),
            evidence_types_used=(),
            domain_profile=domain_profile,
            collaboration_workflow=CollaborationWorkflow.RESEARCH_SESSION,
        )

    def build_research_synthesis_payload(
        self,
        goal: str,
        step_findings: list[tuple[str, str, list[str], list[str]]],
        *,
        answer_mode: AnswerMode,
        retrieval_mode: RetrievalMode,
        citation_sources: list[str],
        domain_profile: DomainProfile = DomainProfile.ELECTRONIC_MUSIC,
        workflow_input: WorkflowInput | None = None,
    ) -> PromptPayload:
        """Build a final synthesis prompt from explicit research steps."""
        workflow_input = workflow_input or WorkflowInput()
        finding_blocks: list[str] = []
        for index, (subquestion, answer, sources, warnings) in enumerate(step_findings, start=1):
            finding_blocks.append(
                f"[Step {index}]\n"
                f"Subquestion: {subquestion}\n"
                f"Finding:\n{answer}\n"
                f"Sources:\n{chr(10).join(sources) if sources else 'No sources'}\n"
                f"Warnings:\n{chr(10).join(warnings) if warnings else 'None'}"
            )

        citation_block = "\n".join(citation_sources) if citation_sources else "No evidence sources were retrieved."
        return PromptPayload(
            system_prompt=_build_system_prompt(
                answer_mode,
                domain_profile=domain_profile,
                collaboration_workflow=CollaborationWorkflow.RESEARCH_SESSION,
            ),
            user_prompt=(
                f"Research goal: {goal}\n"
                f"Domain profile: {domain_profile.value}\n"
                f"Answer mode: {answer_mode.value}\n"
                f"Retrieval mode: {retrieval_mode.value}\n\n"
                "Structured workflow context:\n"
                f"{_format_workflow_input(workflow_input)}\n\n"
                "You are synthesizing the final answer from explicit research steps. "
                "Keep local notes, saved answers, and web evidence distinct.\n\n"
                "Available source labels:\n"
                f"{citation_block}\n\n"
                "Research findings:\n"
                f"{chr(10).join(finding_blocks)}\n\n"
                f"{_workflow_instructions(CollaborationWorkflow.RESEARCH_SESSION)}\n"
                f"{_mode_instructions(answer_mode)}\n"
                "Write a final synthesized answer that cites supported claims and labels broader synthesis with [Inference] when needed."
            ),
            answer_mode=answer_mode,
            citation_labels=tuple(_extract_citation_label(source) for source in citation_sources),
            evidence_types_used=(),
            domain_profile=domain_profile,
            collaboration_workflow=CollaborationWorkflow.RESEARCH_SESSION,
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
    saved_index = 1
    import_index = 1
    for chunk in chunks:
        key = (
            chunk.metadata.get("note_title", "Untitled"),
            chunk.metadata.get("source_path", "unknown"),
        )
        if key in seen_sources:
            continue
        seen_sources.add(key)
        is_saved = _is_saved_answer_chunk(chunk)
        is_import = _is_imported_chunk(chunk)
        if is_saved:
            label = f"[Saved {saved_index}]"
            saved_index += 1
        elif is_import:
            label = f"[Import {import_index}]"
            import_index += 1
        else:
            label = f"[Local {local_index}]"
            local_index += 1
        labeled_sources.append(
            f"{label} {chunk.metadata.get('note_title', 'Untitled')} "
            f"({chunk.metadata.get('source_path', 'unknown')})"
        )
        citation_labels.append(label)

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


def _extract_citation_label(source: str) -> str:
    """Extract a stable leading citation label from a formatted source string."""
    if source.startswith("[") and "]" in source:
        return source.split("]", 1)[0] + "]"
    return source


def _build_system_prompt(
    answer_mode: AnswerMode,
    *,
    domain_profile: DomainProfile,
    collaboration_workflow: CollaborationWorkflow,
    framework_text: str = "",
) -> str:
    common = (
        "You are a careful research assistant for an Obsidian vault. "
        "Keep local-note evidence and external web evidence distinct. "
        "Use citation labels exactly as provided when referencing evidence."
    )
    domain_block = (
        " Default domain: electronic music production and collaboration. "
        "Work comfortably with genre and style fit, BPM and groove, arrangement sections, "
        "tension and release, layering, drum programming, bass design, transitions, "
        "references, mood, energy, atmosphere, and production workflow. "
        "If the user asks something outside music, answer normally without forcing the music frame."
    )
    workflow_block = f" Active collaboration workflow: {collaboration_workflow.value}."
    if answer_mode == AnswerMode.STRICT:
        prompt = (
            common
            + domain_block
            + workflow_block
            + " Strict mode: use only retrieved evidence, refuse when evidence is insufficient, "
            "and do not present unsupported claims as facts."
        )
    elif answer_mode == AnswerMode.EXPLORATORY:
        prompt = (
            common
            + domain_block
            + workflow_block
            + " Exploratory mode: you may synthesize and infer beyond direct evidence, but label "
            "those parts with [Inference] and keep supported claims cited."
        )
    else:
        prompt = (
            common
            + domain_block
            + workflow_block
            + " Balanced mode: prioritize retrieved evidence, allow limited synthesis, and label any "
            "beyond-evidence reasoning with [Inference]."
        )

    if framework_text:
        prompt += _format_internal_framework_block(framework_text)
    return prompt


def _build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    web_results: list[WebSearchResult],
    retrieval_mode: RetrievalMode,
    answer_mode: AnswerMode,
    local_retrieval_weak: bool,
    domain_profile: DomainProfile,
    collaboration_workflow: CollaborationWorkflow,
    workflow_input: WorkflowInput,
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
        f"Domain profile: {domain_profile.value}\n"
        f"Collaboration workflow: {collaboration_workflow.value}\n"
        f"Retrieval mode: {retrieval_mode.value}\n"
        f"Local retrieval strength: {evidence_strength}\n\n"
        "Structured workflow input:\n"
        f"{_format_workflow_input(workflow_input)}\n\n"
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
        f"{_workflow_instructions(collaboration_workflow)}\n"
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


def _workflow_instructions(collaboration_workflow: CollaborationWorkflow) -> str:
    if collaboration_workflow == CollaborationWorkflow.GENRE_FIT_REVIEW:
        return (
            "Workflow instructions:\n"
            "- Assess likely genre or style fit using the provided evidence first.\n"
            "- Call out mismatches, ambiguities, and missing production cues.\n"
            "- Suggest concrete ways to align the idea more strongly with the target style.\n"
            "- Keep evidence-backed observations distinct from inferred genre judgments."
        )
    if collaboration_workflow == CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE:
        return (
            "Workflow instructions:\n"
            "- Critique the track concept like a constructive electronic music collaborator.\n"
            "- Identify what is working, what feels weak or unclear, and what should be developed next.\n"
            "- Include arrangement, energy, and sound-design directions when they are relevant.\n"
            "- Prefer actionable next steps over generic encouragement."
        )
    if collaboration_workflow == CollaborationWorkflow.ARRANGEMENT_PLANNER:
        return (
            "Workflow instructions:\n"
            "- Turn the idea into a practical section-by-section arrangement plan.\n"
            "- Cover section goals, pacing, transitions, tension and release, and variation.\n"
            "- Make the plan readable for a producer returning to a session later."
        )
    if collaboration_workflow == CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM:
        return (
            "Workflow instructions:\n"
            "- Focus on synth, drum, bass, texture, FX, modulation, layering, and space.\n"
            "- Suggest practical production directions rather than abstract adjectives alone.\n"
            "- Note mix-role implications when helpful."
        )
    if collaboration_workflow == CollaborationWorkflow.RESEARCH_SESSION:
        return (
            "Workflow instructions:\n"
            "- This is a deeper research workflow, not a quick response.\n"
            "- Synthesize broader evidence about style, arrangement, sound design, or references.\n"
            "- Preserve the boundary between retrieved local notes, web evidence, and model inference."
        )
    return (
        "Workflow instructions:\n"
        "- Answer like an electronic music research and collaboration assistant.\n"
        "- Use producer-friendly language and make suggestions actionable.\n"
        "- Do not force structure when the user only needs a direct answer."
    )


def _format_workflow_input(workflow_input: WorkflowInput) -> str:
    values = workflow_input.as_dict()
    if not values:
        return "No additional structured workflow input was provided."
    return "\n".join(f"- {key.replace('_', ' ').title()}: {value}" for key, value in values.items())


def _format_internal_framework_block(framework_text: str) -> str:
    return (
        "\n\nBEGIN INTERNAL CRITIQUE FRAMEWORK\n"
        "Apply the following critique system when evaluating the user's track critique request. "
        "Use it as internal operating guidance, not as evidence or user content.\n\n"
        f"{framework_text.strip()}\n"
        "END INTERNAL CRITIQUE FRAMEWORK"
    )


def _format_local_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant local note context was retrieved."
    parts: list[str] = []
    local_index = 1
    saved_index = 1
    import_index = 1
    for chunk in chunks:
        title = chunk.metadata.get("note_title", "Untitled note")
        source_path = chunk.metadata.get("source_path", "unknown")
        heading_context = chunk.metadata.get("heading_context", "")
        section_line = f" | Section: {heading_context}" if heading_context else ""
        if _is_saved_answer_chunk(chunk):
            source_kind = "Saved answer"
        elif _is_imported_chunk(chunk):
            source_kind = "Imported content"
        else:
            source_kind = "Primary note"
        if chunk.metadata.get("linked_context"):
            context_kind = f"Linked {source_kind.lower()}"
        else:
            context_kind = source_kind
        if _is_saved_answer_chunk(chunk):
            label = f"[Saved {saved_index}]"
            saved_index += 1
        elif _is_imported_chunk(chunk):
            label = f"[Import {import_index}]"
            import_index += 1
        else:
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
    if any(_is_saved_answer_chunk(chunk) for chunk in chunks):
        evidence_types.append("saved_answer")
    if any(_is_imported_chunk(chunk) for chunk in chunks):
        evidence_types.append("imported_content")
    if any(
        not _is_saved_answer_chunk(chunk) and not _is_imported_chunk(chunk)
        for chunk in chunks
    ):
        evidence_types.append("local_note")
    if web_results:
        evidence_types.append("web")
    return tuple(evidence_types)


def _is_saved_answer_chunk(chunk: RetrievedChunk) -> bool:
    return str(chunk.metadata.get("source_kind", "")).strip().lower() == "saved_answer"


def _is_imported_chunk(chunk: RetrievedChunk) -> bool:
    return str(chunk.metadata.get("source_kind", "")).strip().lower() == "imported_content"
