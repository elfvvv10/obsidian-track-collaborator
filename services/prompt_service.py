"""Answer-mode prompt policies and citation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from config import AppConfig
from services.framework_service import FrameworkService
from services.models import (
    AnswerMode,
    ChatMessage,
    CollaborationWorkflow,
    DomainProfile,
    RetrievalMode,
    SessionTask,
    TrackContext,
    WorkflowInput,
)
from services.track_context_service import TrackContextService
from utils import RetrievedChunk, get_logger
from web_search import WebSearchResult


logger = get_logger()


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
        track_context_service_cls: type[TrackContextService] = TrackContextService,
    ) -> None:
        self.config = config
        self.framework_service = framework_service_cls(config)
        self.track_context_service = track_context_service_cls(config)

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
        track_id: str | None = None,
        use_track_context: bool = False,
        track_context: TrackContext | None = None,
        recent_conversation: list[ChatMessage] | None = None,
        current_tasks: list[SessionTask] | None = None,
        web_query_used: str = "",
        web_query_strategy: str = "raw_question",
        web_alignment_note: str = "",
    ) -> PromptPayload:
        citation_sources, citation_labels = build_citation_sources(chunks, web_results)
        evidence_types_used = _build_evidence_types(chunks, web_results)
        workflow_input = workflow_input or WorkflowInput()
        recent_conversation = recent_conversation or []
        current_tasks = current_tasks or []
        framework_text = self.framework_service.get_framework_text(collaboration_workflow, domain_profile)
        track_context_text = ""
        if use_track_context and track_id and track_context is not None:
            track_context_text = self._format_track_context(track_context)
        else:
            legacy_track_context = self.track_context_service.get_track_context(
                collaboration_workflow,
                workflow_input.track_context_path,
            )
            track_context_text = legacy_track_context.prompt_block
        critique_instructions = self._format_critique_instructions(
            collaboration_workflow,
            track_context,
            chunks,
            workflow_input,
        )
        system_prompt = _build_system_prompt(
            answer_mode,
            domain_profile=domain_profile,
            collaboration_workflow=collaboration_workflow,
            framework_text=framework_text,
            critique_instructions=critique_instructions,
            track_context_text=track_context_text,
            recent_conversation=recent_conversation,
            current_tasks=current_tasks,
        )
        if self.config.framework_debug:
            logger.info(
                "Prompt internals: workflow=%s framework_injected=%s track_context_injected=%s tasks_injected=%s conversation_injected=%s.",
                collaboration_workflow.value,
                _FRAMEWORK_BLOCK_START in system_prompt,
                _TRACK_CONTEXT_BLOCK_START in system_prompt,
                _CURRENT_TASKS_BLOCK_START in system_prompt,
                _RECENT_CONVERSATION_BLOCK_START in system_prompt,
            )
        return PromptPayload(
            system_prompt=system_prompt,
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

    def _format_track_context(self, track_context: TrackContext) -> str:
        """Format YAML-backed track context for internal prompt injection."""
        lines = [
            "Use this as internal track-state guidance for continuity, prioritization, and finish-oriented advice. "
            "Do not treat it as evidence or a citation source.",
            "",
            "Track context summary:",
            f"- Track Id: {track_context.track_id}",
        ]
        optional_fields = (
            ("Track Name", track_context.track_name),
            ("Genre", track_context.genre),
            ("BPM", track_context.bpm),
            ("Key", track_context.key),
            ("Current Stage", track_context.current_stage),
            ("Current Problem", track_context.current_problem),
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
        return "\n".join(lines)

    def _format_critique_instructions(
        self,
        collaboration_workflow: CollaborationWorkflow,
        track_context: TrackContext | None,
        chunks: list[RetrievedChunk],
        workflow_input: WorkflowInput,
    ) -> str:
        """Return structured critique instructions when critique mode is active."""
        if collaboration_workflow != CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE:
            return ""
        arrangement_summary = _summarize_arrangement_evidence(chunks)
        track_context_guidance = (
            "Use Track Context only for long-term track identity and current production state, "
            "such as genre, BPM, vibe, references, current stage, current problem, known issues, and goals."
            if track_context is not None
            else "If Track Context is unavailable, infer the track identity only from the user's question and retrieved evidence."
        )
        arrangement_guidance = (
            "Arrangement evidence is available. Treat it as the structural representation of the track over time. "
            "Analyze the track section by section where useful, and refer to section names and bar ranges when practical. "
            "Focus on arrangement and energy flow, transitions, pacing, overlong sections, weak drops, static loops, "
            "and how groove, bass, and key elements evolve across sections."
            if arrangement_summary
            else "Structured arrangement evidence is not available. Fall back to a higher-level critique and be explicit "
            "when section-level judgments are limited."
        )
        workflow_context_guidance = (
            "Use any workflow arrangement notes as supporting context, but do not confuse them with structured arrangement evidence."
            if workflow_input.arrangement_notes and workflow_input.arrangement_notes.strip()
            else "If workflow context includes arrangement hints, use them as supporting context rather than as a substitute for evidence."
        )
        lines = [
            "You are acting as a professional electronic music producer giving structured track critique.",
            "",
            track_context_guidance,
            arrangement_guidance,
            workflow_context_guidance,
            "Prioritize the most important weaknesses and opportunities rather than listing every possible issue.",
            "Be specific to this track. Avoid generic filler advice.",
            "When relevant, assess genre/style fit against the track identity and reference tracks.",
            "For each major issue, explain why it matters and what to change in practical production terms.",
            "",
            "Respond using these headings exactly:",
            "- Overall Assessment",
            "- Arrangement / Energy Flow",
            "- Genre / Style Fit",
            "- Groove / Bass / Element Evolution",
            "- Priority Issues",
            "- Recommended Next Changes",
        ]
        if arrangement_summary:
            lines.extend(
                [
                    "",
                    "Arrangement evidence available:",
                    arrangement_summary,
                ]
            )
        return "\n".join(lines)

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
    reference_index = 1
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
        is_reference = _is_reference_chunk(chunk)
        if is_saved:
            label = f"[Saved {saved_index}]"
            saved_index += 1
        elif is_import:
            label = f"[Import {import_index}]"
            import_index += 1
        elif is_reference:
            label = f"[Ref {reference_index}]"
            reference_index += 1
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
    critique_instructions: str = "",
    track_context_text: str = "",
    recent_conversation: list[ChatMessage] | None = None,
    current_tasks: list[SessionTask] | None = None,
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

    if critique_instructions:
        prompt += f"\n\n{critique_instructions.strip()}"

    chat_task_enabled = collaboration_workflow in {
        CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
        CollaborationWorkflow.ARRANGEMENT_PLANNER,
        CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM,
    }
    if framework_text and _FRAMEWORK_BLOCK_START not in prompt:
        prompt += _format_internal_framework_block(framework_text)
    if track_context_text and _TRACK_CONTEXT_BLOCK_START not in prompt:
        prompt += _format_internal_track_context_block(track_context_text)
    if chat_task_enabled:
        tasks_block = _format_current_tasks_block(current_tasks or [])
        conversation_block = _format_recent_conversation_block(recent_conversation or [])
        if tasks_block and _CURRENT_TASKS_BLOCK_START not in prompt:
            prompt += tasks_block
        if conversation_block and _RECENT_CONVERSATION_BLOCK_START not in prompt:
            prompt += conversation_block
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
            f"{_music_collaboration_instruction_block()}\n"
            "- Critique the track concept like a constructive electronic music collaborator.\n"
            "- Treat Track Context as long-term track identity and current production state, and treat arrangement evidence as structural track information over time.\n"
            "- Identify what is working, what feels weak or unclear, and what should be developed next.\n"
            "- Include arrangement, energy, and sound-design directions when they are relevant.\n"
            "- When arrangement evidence is available, analyze it section by section and refer to section names or bar ranges when practical.\n"
            "- Prefer section-aware critique over generic advice when the arrangement context supports it.\n"
            "- Organize the critique with these headings exactly: Overall Assessment, Arrangement / Energy Flow, Genre / Style Fit, Groove / Bass / Element Evolution, Priority Issues, Recommended Next Changes.\n"
            "- For major critique points, explain the issue, why it matters, how to implement the change, a minimal first pass to try quickly, and what to listen for afterward.\n"
            "- Prefer practical studio actions over abstract commentary, including arrangement moves, automation moves, sound-design moves, drum or percussion changes, bass or low-end changes, transition-building techniques, and subtraction when useful.\n"
            "- Prioritize the highest-impact issues rather than trying to diagnose everything at once.\n"
            "- Prefer actionable next steps over generic encouragement."
        )
    if collaboration_workflow == CollaborationWorkflow.ARRANGEMENT_PLANNER:
        return (
            "Workflow instructions:\n"
            f"{_music_collaboration_instruction_block()}\n"
            "- Turn the idea into a practical section-by-section arrangement plan.\n"
            "- Cover section goals, pacing, transitions, tension and release, and variation.\n"
            "- For major suggestions, explain why the change matters, how to implement it in practical production terms, a minimal first pass, and what to listen for afterward.\n"
            "- Use concrete production language around transitions, automation, removal, low-end control, percussion movement, and contrast between sections.\n"
            "- Make the plan readable for a producer returning to a session later."
        )
    if collaboration_workflow == CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM:
        return (
            "Workflow instructions:\n"
            f"{_music_collaboration_instruction_block()}\n"
            f"{_sound_design_structured_output_block()}\n"
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


def _music_collaboration_instruction_block() -> str:
    return (
        "- Start with a direct answer to the user's music question. Do not open with framing language such as 'Based on the provided context' or 'From the sources'.\n"
        "- Use retrieved material to support, constrain, or refine the answer after the direct answer, not to replace it or structure the response around it.\n"
        "- Ignore weak, tangential, or loosely related sources instead of forcing them into the response.\n"
        "- Every meaningful suggestion must include how to do it in practical production terms, such as MIDI patterns, synth settings, arrangement moves, automation moves, drum edits, bass edits, transition techniques, or subtraction.\n"
        "- Do not stop at abstract advice. If you name a change, explain the execution steps or give a concrete example the producer can try immediately.\n"
        "- Stay anchored to the stated genre, style, and workflow context. If a genre is present, prioritize genre-native techniques first and avoid drifting into other genres.\n"
        "- Treat cross-genre or adjacent-genre ideas as optional variations and label them clearly as optional.\n"
        "- If the user asks for ideas or options, provide multiple concrete, usable ideas rather than summarizing sources or staying conceptual."
    )


def _sound_design_structured_output_block() -> str:
    return (
        "- For this workflow, structure the answer exactly as: Quick Answer, Production Recipes, and Optional Variations when relevant.\n"
        "- Quick Answer: 1-2 lines, no fluff, and no source-led framing.\n"
        "- Production Recipes: provide 2-4 concrete ideas.\n"
        "- For each production recipe, include these headings exactly: Name, Groove / MIDI, Sound Design, How to Build It, Where to Use It.\n"
        "- Groove / MIDI must include timing guidance and note-length guidance.\n"
        "- Sound Design must cover oscillator choice, filter direction, envelope behavior, and saturation or distortion when relevant.\n"
        "- How to Build It must be step-by-step, numbered, and written as DAW-level actions.\n"
        "- Where to Use It must name the arrangement context, such as intro, breakdown, build, or drop.\n"
        "- Do not open with phrases such as 'Based on the provided context' or 'According to sources'.\n"
        "- Do not produce generic advice or filler phrases such as 'experiment with', 'try different', or 'remember that'.\n"
        "- Keep the output actionable, specific, and genre-grounded. Prioritize progressive-house conventions when the genre context points there.\n"
        "- Core recipes must be musically plausible for the requested genre or style and should prioritize genre-common archetypes first.\n"
        "- Avoid novelty, gimmicks, unusual hybrids, or structurally inappropriate ideas unless they are clearly labeled as optional variations.\n"
        "- Do not confuse bassline design, percussion fills, drum tricks, and unrelated sound categories in the main recipe list.\n"
        "- Weakly related or cross-genre retrieved material must not become core recommendations.\n"
        "- If cross-genre material is genuinely useful, include it only as optional inspiration or an optional variation.\n"
        "- Do not let imported or cross-genre source material distort the main recipe list.\n"
        "- Prefer recommendations that a knowledgeable producer in the requested genre would recognize as sensible starting points.\n"
        "- Reject arbitrary or musically implausible ideas even if a retrieved source mentions them.\n"
        "- Cross-genre ideas are allowed only as clearly labeled optional variations."
    )


def _format_workflow_input(workflow_input: WorkflowInput) -> str:
    values = workflow_input.as_dict()
    if not values:
        return "No additional structured workflow input was provided."
    return "\n".join(f"- {key.replace('_', ' ').title()}: {value}" for key, value in values.items())


def _format_internal_framework_block(framework_text: str) -> str:
    return (
        f"\n\n{_FRAMEWORK_BLOCK_START}\n"
        "Apply the following critique system when evaluating the user's track critique request. "
        "Use it as internal operating guidance, not as evidence or user content.\n\n"
        f"{framework_text.strip()}\n"
        f"{_FRAMEWORK_BLOCK_END}"
    )


_FRAMEWORK_BLOCK_START = "BEGIN INTERNAL CRITIQUE FRAMEWORK"
_FRAMEWORK_BLOCK_END = "END INTERNAL CRITIQUE FRAMEWORK"
_TRACK_CONTEXT_BLOCK_START = "BEGIN INTERNAL TRACK CONTEXT"
_TRACK_CONTEXT_BLOCK_END = "END INTERNAL TRACK CONTEXT"
_CURRENT_TASKS_BLOCK_START = "BEGIN CURRENT TASKS"
_CURRENT_TASKS_BLOCK_END = "END CURRENT TASKS"
_RECENT_CONVERSATION_BLOCK_START = "BEGIN RECENT CONVERSATION"
_RECENT_CONVERSATION_BLOCK_END = "END RECENT CONVERSATION"


def _format_internal_track_context_block(track_context_text: str) -> str:
    return (
        f"\n\n{_TRACK_CONTEXT_BLOCK_START}\n"
        f"{track_context_text.strip()}\n"
        f"{_TRACK_CONTEXT_BLOCK_END}"
    )


def _format_current_tasks_block(current_tasks: list[SessionTask]) -> str:
    if not current_tasks:
        return ""
    ordered_tasks = sorted(
        current_tasks,
        key=lambda task: (task.status != "open", task.created_at, task.id),
    )
    lines: list[str] = []
    for task in ordered_tasks:
        if not task.text.strip():
            continue
        checkbox = "[ ]" if task.status == "open" else "[x]"
        line = f"{checkbox} {task.text.strip()}"
        if task.notes.strip():
            line += f" ({task.notes.strip()})"
        lines.append(line)
    if not lines:
        return ""
    return (
        f"\n\n{_CURRENT_TASKS_BLOCK_START}\n"
        "Use these as internal execution priorities for the current music session. "
        "Do not treat them as evidence or citation sources.\n\n"
        + "\n".join(lines)
        + f"\n{_CURRENT_TASKS_BLOCK_END}"
    )


def _format_recent_conversation_block(recent_conversation: list[ChatMessage]) -> str:
    if not recent_conversation:
        return ""
    lines = [
        f"{message.role.title()}: {message.content.strip()}"
        for message in recent_conversation[-8:]
        if message.content.strip()
    ]
    if not lines:
        return ""
    return (
        f"\n\n{_RECENT_CONVERSATION_BLOCK_START}\n"
        "Use this as recent collaboration continuity for the current session. "
        "Do not treat it as evidence or citation sources.\n\n"
        + "\n".join(lines)
        + f"\n{_RECENT_CONVERSATION_BLOCK_END}"
    )


def _format_local_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant local note context was retrieved."
    parts: list[str] = []
    local_index = 1
    reference_index = 1
    saved_index = 1
    import_index = 1
    for chunk in chunks:
        title = chunk.metadata.get("note_title", "Untitled note")
        source_path = chunk.metadata.get("source_path", "unknown")
        heading_context = chunk.metadata.get("heading_context", "")
        section_line = f" | Section: {heading_context}" if heading_context else ""
        arrangement_track_name = str(chunk.metadata.get("arrangement_track_name", "")).strip()
        arrangement_section_name = str(chunk.metadata.get("arrangement_section_name", "")).strip()
        video_title = str(chunk.metadata.get("video_title", "")).strip()
        video_section_title = str(chunk.metadata.get("video_section_title", "")).strip()
        video_start_time = str(chunk.metadata.get("video_start_time", "")).strip()
        video_end_time = str(chunk.metadata.get("video_end_time", "")).strip()
        if _is_saved_answer_chunk(chunk):
            source_kind = "Saved answer"
        elif _is_video_chunk(chunk):
            source_kind = "Video reference evidence"
        elif _is_imported_chunk(chunk):
            source_kind = "Imported reference evidence"
        elif _is_arrangement_chunk(chunk):
            source_kind = "Arrangement reference"
        elif _is_reference_chunk(chunk):
            source_kind = "Reference evidence"
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
        elif _is_reference_chunk(chunk):
            label = f"[Ref {reference_index}]"
            reference_index += 1
        else:
            label = f"[Local {local_index}]"
            local_index += 1
        arrangement_lines = []
        if arrangement_track_name:
            arrangement_lines.append(f"Arrangement Track: {arrangement_track_name}")
        if arrangement_section_name and arrangement_section_name.lower() != "arrangement overview":
            arrangement_lines.append(f"Arrangement Section: {arrangement_section_name}")
        if video_title:
            arrangement_lines.append(f"Video Title: {video_title}")
        if video_section_title and video_section_title.lower() != "overview":
            arrangement_lines.append(f"Video Section: {video_section_title}")
        if video_start_time or video_end_time:
            arrangement_lines.append(f"Video Timestamp: {video_start_time or '?'} - {video_end_time or '?'}")
        arrangement_block = "\n".join(arrangement_lines)
        if arrangement_block:
            arrangement_block = f"{arrangement_block}\n"
        parts.append(
            f"{label}\n"
            f"Type: {context_kind}\n"
            f"Title: {title}{section_line}\n"
            f"Path: {source_path}\n"
            f"{arrangement_block}"
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


def _is_arrangement_chunk(chunk: RetrievedChunk) -> bool:
    return str(chunk.metadata.get("source_type", "")).strip().lower() == "track_arrangement"


def _is_video_chunk(chunk: RetrievedChunk) -> bool:
    return str(chunk.metadata.get("source_type", "")).strip().lower() == "youtube_video"


def _is_reference_chunk(chunk: RetrievedChunk) -> bool:
    content_category = str(chunk.metadata.get("content_category", "")).strip().lower()
    content_scope = str(chunk.metadata.get("content_scope", "")).strip().lower()
    if content_category == "curated_knowledge":
        return True
    return content_scope == "knowledge" and not _is_imported_chunk(chunk) and not _is_saved_answer_chunk(chunk)


def _summarize_arrangement_evidence(chunks: list[RetrievedChunk]) -> str:
    """Return a short section-aware arrangement summary for critique instructions."""
    section_lines: list[str] = []
    seen_sections: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not _is_arrangement_chunk(chunk):
            continue
        track_name = str(chunk.metadata.get("arrangement_track_name", "")).strip()
        section_name = str(chunk.metadata.get("arrangement_section_name", "")).strip()
        if not section_name or section_name.lower() == "arrangement overview":
            continue
        key = (track_name, section_name)
        if key in seen_sections:
            continue
        seen_sections.add(key)
        section_parts = [section_name]
        heading_context = str(chunk.metadata.get("heading_context", "")).strip()
        bar_match = re.search(r"\bBars:\s*(\d+)\s*-\s*(\d+)\b", chunk.text)
        if bar_match:
            section_parts.append(f"Bars {bar_match.group(1)}-{bar_match.group(2)}")
        elif heading_context and heading_context != section_name:
            section_parts.append(heading_context)
        energy = chunk.metadata.get("arrangement_energy")
        if energy not in ("", None):
            section_parts.append(f"Energy {energy}")
        section_lines.append(f"- {' | '.join(section_parts)}")
        if len(section_lines) >= 6:
            break
    return "\n".join(section_lines)
