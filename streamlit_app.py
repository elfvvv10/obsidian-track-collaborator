"""Local Streamlit UI for the Obsidian RAG assistant."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import streamlit as st

from config import AppConfig, load_config
from llm import list_available_chat_models
from services.ingestion_service import IngestionService
from services.index_service import IndexService
from services.music_workflow_service import MusicWorkflowService
from services.models import (
    AnswerMode,
    ChatMessage,
    CollaborationWorkflow,
    IngestionRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    ResearchResponse,
    RetrievalMode,
    RetrievalScope,
    SessionTask,
    WorkflowInput,
    WorkflowMode,
)
from services.query_service import QueryService
from services.research_service import ResearchService
from services.track_selector_service import TrackSelectorService, selected_track_index, selected_track_path
from services.ui_session_helpers import current_track_summary, debug_query_summary, suggestion_groups
from utils import RetrievalFilters, RetrievalOptions, current_timestamp


st.set_page_config(page_title="Electronic Music Research Assistant", layout="wide")


def main() -> None:
    """Render the Streamlit app."""
    st.title("Electronic Music Research and Collaboration Assistant")
    st.caption("Local-first collaboration for genre fit, track critique, arrangement planning, sound design, and research.")

    try:
        base_config = load_config()
    except Exception as exc:
        st.error(str(exc))
        return

    _init_session_state(base_config)
    ui_config = _config_from_session(base_config)
    services = _get_services(ui_config)
    status, status_error = _safe_get_status(services["index_service"])

    _render_sidebar(base_config, status, status_error, services["query_service"])

    ask_tab, ingest_tab, index_tab, settings_tab = st.tabs(
        ["Ask", "Ingest", "Index", "Settings / Debug"]
    )

    with ask_tab:
        _render_ask_tab(
            ui_config,
            services["query_service"],
            services["research_service"],
            services["music_workflow_service"],
            status,
        )

    with ingest_tab:
        _render_ingest_tab(services["ingestion_service"])

    with index_tab:
        _render_index_tab(services["index_service"], status)

    with settings_tab:
        _render_settings_tab(base_config, status, status_error)


def _get_services(config: AppConfig) -> dict[str, object]:
    """Construct shared services for the current UI config."""
    return {
        "query_service": QueryService(config),
        "research_service": ResearchService(config),
        "index_service": IndexService(config),
        "ingestion_service": IngestionService(config),
        "music_workflow_service": MusicWorkflowService(config),
    }


def _safe_get_status(index_service: IndexService) -> tuple[IndexResponse | None, str | None]:
    try:
        return index_service.get_status(), None
    except Exception as exc:
        return None, str(exc)


def _render_sidebar(
    config: AppConfig,
    status: IndexResponse | None,
    status_error: str | None,
    query_service: QueryService,
) -> None:
    with st.sidebar:
        st.header("Retrieval Settings")
        st.caption("These stay in the background while the composer remains the main workflow.")
        with st.expander("Adjust Filters and Modes", expanded=False):
            with st.form("sidebar_controls", clear_on_submit=False, enter_to_submit=False):
                st.text_input(
                    "Folder filter",
                    key="folder_filter",
                    help="Only search notes from this vault-relative folder.",
                )
                st.text_input(
                    "Path contains",
                    key="path_filter",
                    help="Only search notes whose path contains this text.",
                )
                st.text_input(
                    "Tag filter",
                    key="tag_filter",
                    help="Only search notes with this tag.",
                )
                st.number_input(
                    "Top-k",
                    min_value=1,
                    max_value=20,
                    help="How many final chunks to use when answering.",
                    key="top_k",
                )
                st.checkbox(
                    "Enable reranking",
                    key="enable_reranking",
                )
                st.checkbox(
                    "Include linked-note context",
                    key="include_linked",
                )
                st.checkbox(
                    "Auto-save answers",
                    key="auto_save",
                )
                st.selectbox(
                    "Retrieval mode",
                    options=[mode.value for mode in RetrievalMode],
                    help="Choose local-only, automatic web fallback, or always-on hybrid web use.",
                    key="retrieval_mode",
                )
                st.selectbox(
                    "Answer mode",
                    options=[mode.value for mode in AnswerMode],
                    help=(
                        "Strict: retrieved evidence only. "
                        "Balanced: evidence first with limited reasoning. "
                        "Exploratory: evidence plus broader synthesis and labeled inference."
                    ),
                    key="answer_mode",
                )
                st.form_submit_button("Apply Retrieval Settings", use_container_width=True)
        st.caption("Retrieval scope is chosen in the Ask tab so it stays close to the current prompt.")

        st.divider()
        st.subheader("YAML Track Context")
        st.caption("Primary editable Track Context used for prompts, retrieval rewriting, critique mode, and suggested updates.")
        st.text_input(
            "Track ID",
            key="track_context_track_id",
            help="Persistent YAML track context identifier for the new editable flow.",
        )
        st.checkbox(
            "Use Track Context",
            key="use_track_context",
            help="Enable YAML-backed track context for direct asks and research workflows.",
        )
        track_id = st.session_state.get("track_context_track_id", "").strip()
        if st.session_state.get("use_track_context") and track_id:
            context = query_service.track_context_service.load_or_create(track_id)
            with st.expander("Edit YAML Track Context", expanded=False):
                with st.form("track_context_editor", clear_on_submit=False, enter_to_submit=False):
                    st.caption("Core Track Info")
                    core_col_left, core_col_right = st.columns(2)
                    core_col_left.text_input(
                        "Name",
                        value=context.track_name or "",
                        key="track_context_track_name",
                    )
                    core_col_right.text_input("Genre", value=context.genre or "", key="track_context_genre")
                    detail_col_left, detail_col_right = st.columns(2)
                    detail_col_left.text_input(
                        "BPM",
                        value="" if context.bpm is None else str(context.bpm),
                        key="track_context_bpm",
                    )
                    detail_col_right.text_input("Key", value=context.key or "", key="track_context_key")
                    st.caption("Session Focus")
                    focus_col_left, focus_col_right = st.columns(2)
                    focus_col_left.selectbox(
                        "Workflow",
                        options=_TRACK_CONTEXT_WORKFLOW_MODES,
                        index=_TRACK_CONTEXT_WORKFLOW_MODES.index(context.workflow_mode),
                        key="track_context_workflow_mode",
                    )
                    stage_options = [""] + _TRACK_CONTEXT_CURRENT_STAGES
                    current_stage = context.current_stage or ""
                    focus_col_right.selectbox(
                        "Stage",
                        options=stage_options,
                        index=stage_options.index(current_stage),
                        key="track_context_current_stage",
                    )
                    st.text_input(
                        "Section",
                        value=context.current_section or "",
                        key="track_context_current_section",
                    )
                    st.text_input(
                        "Vibe (comma separated)",
                        value=", ".join(context.vibe),
                        key="track_context_vibe",
                    )
                    st.caption("References and Working Notes")
                    st.text_area(
                        "Reference Tracks (one per line)",
                        value="\n".join(context.reference_tracks),
                        key="track_context_reference_tracks",
                        height=70,
                    )
                    st.text_area(
                        "Known Issues (one per line)",
                        value="\n".join(context.known_issues),
                        key="track_context_known_issues",
                        height=70,
                    )
                    st.text_area(
                        "Goals (one per line)",
                        value="\n".join(context.goals),
                        key="track_context_goals",
                        height=70,
                    )
                    st.text_area(
                        "Notes (one per line)",
                        value="\n".join(context.notes),
                        key="track_context_notes",
                        height=90,
                    )
                    saved = st.form_submit_button("Save Track Context", use_container_width=True)
                if saved:
                    query_service.track_context_service.update_fields(
                        track_id,
                        {
                            "track_name": st.session_state["track_context_track_name"],
                            "genre": st.session_state["track_context_genre"],
                            "bpm": st.session_state["track_context_bpm"],
                            "key": st.session_state["track_context_key"],
                            "workflow_mode": st.session_state["track_context_workflow_mode"],
                            "current_stage": st.session_state["track_context_current_stage"],
                            "current_section": st.session_state["track_context_current_section"],
                            "vibe": _split_csv(st.session_state["track_context_vibe"]),
                            "reference_tracks": _split_lines(
                                st.session_state["track_context_reference_tracks"]
                            ),
                            "known_issues": _split_lines(
                                st.session_state["track_context_known_issues"]
                            ),
                            "goals": _split_lines(st.session_state["track_context_goals"]),
                            "notes": _split_lines(st.session_state["track_context_notes"]),
                        },
                    )
                    st.success("Track context saved.")
        elif st.session_state.get("use_track_context"):
            st.caption("Enter a Track ID to load or create a persistent YAML track context.")

        st.divider()
        st.subheader("App Readiness")
        if status is None:
            st.warning("Status is currently unavailable.")
            if status_error:
                st.caption(status_error)
        else:
            if status.ready:
                st.success("Index is ready for questions.")
            elif status.total_chunks_stored == 0:
                st.info("No indexed chunks yet. Build the index first.")
            elif not status.index_compatible:
                st.warning("Index exists but needs a rebuild.")
            else:
                st.info("Index is present, but readiness is incomplete.")

            if status.ollama_reachable is False:
                st.error(status.ollama_status_message)
            elif status.ollama_reachable is True:
                st.caption(status.ollama_status_message)
            else:
                st.caption("Ollama reachability has not been checked yet.")

            st.caption(f"Stored chunks: {status.total_chunks_stored}")
            st.caption(f"Chat model: {config.ollama_chat_model}")
            st.caption(f"Embedding model: {config.ollama_embedding_model}")


def _render_ask_tab(
    config: AppConfig,
    query_service: QueryService,
    research_service: ResearchService,
    music_workflow_service: MusicWorkflowService,
    status: IndexResponse | None,
) -> None:
    if st.session_state.get("reset_ask_form"):
        for key in (
            "question_input",
            "submitted_message",
            "save_title",
            "workflow_genre",
            "workflow_bpm",
            "workflow_references",
            "workflow_mood",
            "workflow_arrangement_notes",
            "workflow_instrumentation",
            "workflow_sound_palette",
            "workflow_energy_goal",
            "workflow_track_length",
            "workflow_role_of_key_elements",
            "workflow_track_context_path",
            "workflow_track_selector",
            "track_context_track_id",
            "track_context_track_name",
            "track_context_genre",
            "track_context_bpm",
            "track_context_key",
            "track_context_current_section",
            "track_context_vibe",
            "track_context_reference_tracks",
            "track_context_known_issues",
            "track_context_goals",
            "track_context_notes",
            "new_task_text",
            "new_task_notes",
        ):
            st.session_state[key] = ""
        st.session_state["collaboration_workflow"] = CollaborationWorkflow.GENERAL_ASK.value
        st.session_state["workflow_mode"] = WorkflowMode.DIRECT.value
        st.session_state["use_track_context"] = True
        st.session_state["track_context_workflow_mode"] = "general"
        st.session_state["track_context_current_stage"] = ""
        st.session_state["max_subquestions"] = 3
        st.session_state["chat_messages"] = []
        st.session_state["session_tasks"] = []
        st.session_state["last_query_response"] = None
        st.session_state["last_question"] = ""
        st.session_state["active_chat_model"] = _DEFAULT_ACTIVE_CHAT_MODEL
        st.session_state["active_chat_model_select"] = _DEFAULT_ACTIVE_CHAT_MODEL
        st.session_state["active_chat_model_input"] = _DEFAULT_ACTIVE_CHAT_MODEL
        st.session_state["reset_ask_form"] = False

    if status is not None:
        if status.total_chunks_stored == 0:
            st.info("Build the index in the Index tab before asking questions.")
        elif not status.index_compatible:
            st.warning("The local index is out of date. Rebuild it from the Index tab.")

    selected_workflow = CollaborationWorkflow.coerce(st.session_state["collaboration_workflow"])
    chat_workspace_enabled = selected_workflow.value in _CHAT_TASK_WORKFLOWS
    ask_clicked = False
    clear_clicked = False
    question = ""

    answer_mount = None
    chat_detail_mount = None
    available_chat_models, chat_model_discovery_error = list_available_chat_models(config)
    active_track_context = _active_yaml_track_context(query_service)

    main_col, control_col = st.columns([3, 1.4], gap="large")
    with control_col:
        st.markdown("### Workspace Controls")
        st.caption("Workflow, context, reset, and tasks stay close by without interrupting the conversation flow.")
        with st.form("ask_workspace_controls", clear_on_submit=False, enter_to_submit=False):
            workflow_options = [workflow.value for workflow in CollaborationWorkflow]
            selected_workflow_value = st.selectbox(
                "Workflow",
                options=workflow_options,
                format_func=_workflow_label,
                help="Choose a music collaboration workflow or a deeper research session.",
                key="collaboration_workflow",
            )
            selected_workflow = CollaborationWorkflow.coerce(selected_workflow_value)
            st.session_state["workflow_mode"] = (
                WorkflowMode.RESEARCH.value
                if selected_workflow == CollaborationWorkflow.RESEARCH_SESSION
                else WorkflowMode.DIRECT.value
            )
            st.caption(_workflow_help_text(selected_workflow))
            if selected_workflow == CollaborationWorkflow.RESEARCH_SESSION:
                st.number_input(
                    "Max research subquestions",
                    min_value=1,
                    max_value=5,
                    help="Research mode decomposes your request into a small visible set of subquestions.",
                    key="max_subquestions",
                )

            with st.expander("Workflow Context", expanded=selected_workflow != CollaborationWorkflow.RESEARCH_SESSION):
                st.caption("Legacy Markdown Track Context Path")
                st.caption("Optional legacy path-based context. This is separate from the YAML Track Context controls in the sidebar.")
                legacy_tracks = TrackSelectorService().list_tracks(config.obsidian_vault_path)
                track_options = ["None"] + [track["name"] for track in legacy_tracks]
                selected_track = st.selectbox(
                    "Select Track (from vault)",
                    options=track_options,
                    index=selected_track_index(
                        st.session_state.get("workflow_track_context_path", ""),
                        legacy_tracks,
                    ),
                    key="workflow_track_selector",
                    help="Choose any project folder under Projects/ that contains track_context.md to fill the legacy markdown Track Context Path.",
                )
                _apply_legacy_track_selection(selected_track, legacy_tracks)
                st.text_input(
                    "Track Context Path",
                    key="workflow_track_context_path",
                    help="Vault-relative project folder or track_context.md path, for example Projects/Current Tracks/Moonlit Driver or Projects/Current Tracks/Moonlit Driver/track_context.md.",
                )
                st.text_input(
                    "Genre / Style",
                    key="workflow_genre",
                    help="Examples: house, techno, melodic techno, trance, garage, breakbeat.",
                )
                st.text_input(
                    "References",
                    key="workflow_references",
                    help="Artists, tracks, labels, or scenes.",
                )
                st.text_input("BPM / Tempo", key="workflow_bpm")
                st.text_input("Mood / Energy", key="workflow_mood")
                st.text_input("Energy Goal", key="workflow_energy_goal")
                st.text_input("Track Length", key="workflow_track_length")
                st.text_input(
                    "Role of Key Elements",
                    key="workflow_role_of_key_elements",
                    help="What the kick, bass, lead, textures, or vocal should do in the track.",
                )
                st.text_area("Arrangement Notes", key="workflow_arrangement_notes", height=80)
                st.text_area("Instrumentation", key="workflow_instrumentation", height=80)
                st.text_area("Sound Palette", key="workflow_sound_palette", height=80)

            st.markdown("#### Retrieval Scope")
            st.radio(
                "Local retrieval scope",
                options=[RetrievalScope.KNOWLEDGE.value, RetrievalScope.EXTENDED.value],
                format_func=lambda value: (
                    "Knowledge — curated notes + imported reference material"
                    if value == RetrievalScope.KNOWLEDGE.value
                    else "Extended — knowledge + working notes + generated drafts"
                ),
                key="retrieval_scope",
            )
            st.text_input(
                "Optional note title",
                key="save_title",
                help="Override the saved note title and filename slug.",
            )
            if available_chat_models:
                current_model = _resolve_preferred_chat_model_name(
                    st.session_state.get("active_chat_model", _DEFAULT_ACTIVE_CHAT_MODEL),
                    available_chat_models,
                )
                model_options = _dedupe_chat_model_options(available_chat_models)
                if current_model not in model_options:
                    model_options.insert(0, current_model)
                st.selectbox(
                    "Active Chat Model",
                    options=model_options,
                    index=model_options.index(current_model),
                    key="active_chat_model_select",
                    help="Session-level chat model override for comparing local Ollama models.",
                )
            else:
                st.text_input(
                    "Active Chat Model",
                    key="active_chat_model_input",
                    help="Session-level chat model override when live model discovery is unavailable.",
                )
            workspace_updated = st.form_submit_button("Update Workspace", use_container_width=True)

        if workspace_updated:
            selected_chat_model = (
                st.session_state.get("active_chat_model_select", "").strip()
                if available_chat_models
                else st.session_state.get("active_chat_model_input", "").strip()
            ) or _DEFAULT_ACTIVE_CHAT_MODEL
            if available_chat_models:
                selected_chat_model = _resolve_preferred_chat_model_name(
                    selected_chat_model,
                    available_chat_models,
                )
            st.session_state["active_chat_model"] = selected_chat_model

        if st.session_state["retrieval_scope"] == RetrievalScope.KNOWLEDGE.value:
            st.caption(
                "Knowledge searches curated Knowledge folders plus indexed imported reference material such as webpages and YouTube transcripts."
            )
        else:
            st.caption(
                "Extended searches Knowledge, plus indexed working notes, Drafts, and Research Sessions."
            )
        if available_chat_models:
            st.caption(f"Current session model: `{st.session_state['active_chat_model']}`")
        else:
            st.caption(f"Current session model: `{st.session_state['active_chat_model']}`")
            if chat_model_discovery_error:
                st.caption(f"Model discovery unavailable: {chat_model_discovery_error}")

        if st.button("Reset Session", use_container_width=True):
            clear_clicked = True
        st.caption("Reset Session clears the current chat, tasks, and composer/workflow context for this session.")

        if st.session_state.get("last_query_response") and st.session_state["last_query_response"].has_saved:
            st.success(f"Saved to {st.session_state['last_query_response'].saved_path}")
        else:
            st.caption(
                f"This workflow saves by default to `{music_workflow_service.default_save_path(selected_workflow)}`."
            )

        _render_task_panel()
        chat_detail_mount = st.container()

    chat_workspace_enabled = selected_workflow.value in _CHAT_TASK_WORKFLOWS

    with main_col:
        _render_current_track_summary(active_track_context)
        if chat_workspace_enabled:
            st.markdown("### Session Chat")
            st.caption("Read the latest turn above, then reply immediately below to keep the collaboration moving.")
            _render_chat_debug_panel(selected_workflow.value)
            with st.container(border=True):
                _render_chat_history()
                if not st.session_state.get("chat_messages"):
                    st.caption("No messages yet. Start with a critique, arrangement, or sound-design question to open the session.")

            with st.container(border=True):
                st.markdown("#### Reply")
                st.caption("Type your next follow-up here. The newest messages stay directly above the composer.")
                with st.form("chat_composer", clear_on_submit=False, enter_to_submit=False):
                    question = st.text_area(
                        "Message",
                        key="question_input",
                        placeholder="Ask a follow-up, request implementation help, or refine the current idea.",
                        height=110,
                        label_visibility="collapsed",
                    )
                    ask_clicked = st.form_submit_button(
                        "Send",
                        type="primary",
                        use_container_width=True,
                        on_click=_submit_question_from_input,
                    )
        else:
            st.markdown("### Composer")
            st.caption("This workflow keeps a simpler ask/answer flow. Music chat continuity becomes active for critique, arrangement, and sound design workflows.")
            with st.container(border=True):
                with st.form("ask_composer", clear_on_submit=False, enter_to_submit=False):
                    question = st.text_area(
                        "Prompt",
                        key="question_input",
                        placeholder="Describe the track idea, style target, arrangement problem, sound design goal, or research question.",
                        height=120,
                        label_visibility="collapsed",
                    )
                    ask_clicked = st.form_submit_button(
                        "Send",
                        type="primary",
                        use_container_width=True,
                        on_click=_submit_question_from_input,
                    )

        answer_mount = st.container()

    if clear_clicked:
        st.session_state["reset_ask_form"] = True
        st.rerun()

    if ask_clicked:
        submitted_question = st.session_state.get("submitted_message", "").strip()
        if not submitted_question:
            st.warning("Enter a question before asking.")
        else:
            try:
                chat_messages = list(st.session_state["chat_messages"])
                if chat_workspace_enabled:
                    chat_messages.append(
                        ChatMessage(role="user", content=submitted_question, created_at=current_timestamp())
                    )
                request = QueryRequest(
                    question=submitted_question,
                    filters=_current_filters(),
                    options=_current_options(),
                    auto_save=st.session_state["auto_save"],
                    save_title=st.session_state["save_title"].strip() or None,
                    retrieval_scope=st.session_state["retrieval_scope"],
                    retrieval_mode=st.session_state["retrieval_mode"],
                    answer_mode=st.session_state["answer_mode"],
                    collaboration_workflow=st.session_state["collaboration_workflow"],
                    workflow_input=_current_workflow_input(),
                    track_id=st.session_state["track_context_track_id"].strip() or None,
                    use_track_context=st.session_state["use_track_context"],
                    chat_model_override=st.session_state["active_chat_model"],
                    recent_conversation=_recent_conversation_for_prompt(
                        chat_messages[:-1] if chat_workspace_enabled else [],
                        st.session_state["collaboration_workflow"],
                    ),
                    current_tasks=_tasks_for_prompt(
                        st.session_state["session_tasks"],
                        st.session_state["collaboration_workflow"],
                    ),
                )
                if st.session_state["collaboration_workflow"] == CollaborationWorkflow.RESEARCH_SESSION.value:
                    response = research_service.research(
                        ResearchRequest(
                            goal=submitted_question,
                            filters=request.filters,
                            options=request.options,
                            auto_save=request.auto_save,
                            save_title=request.save_title,
                            retrieval_scope=request.retrieval_scope,
                            retrieval_mode=request.retrieval_mode,
                            answer_mode=request.answer_mode,
                            collaboration_workflow=st.session_state["collaboration_workflow"],
                            workflow_input=request.workflow_input,
                            track_id=request.track_id,
                            use_track_context=request.use_track_context,
                            chat_model_override=request.chat_model_override,
                            max_subquestions=st.session_state["max_subquestions"],
                        )
                    )
                else:
                    response = query_service.ask(request)
                if chat_workspace_enabled and not isinstance(response, ResearchResponse):
                    chat_messages.append(
                        ChatMessage(
                            role="assistant",
                            content=response.answer,
                            created_at=current_timestamp(),
                        )
                    )
                    st.session_state["chat_messages"] = chat_messages
                st.session_state["last_query_response"] = response
                st.session_state["last_question"] = submitted_question
                st.session_state["submitted_message"] = ""
                if chat_workspace_enabled and not isinstance(response, ResearchResponse):
                    st.rerun()
            except Exception as exc:
                st.session_state["last_query_response"] = None
                st.session_state["submitted_message"] = ""
                st.error(str(exc))

    response = st.session_state.get("last_query_response")
    if response is None:
        return

    if isinstance(response, ResearchResponse):
        _render_research_response(st.session_state.get("last_question", ""), response, research_service)
        return

    with answer_mount:
        st.markdown("### Latest Answer")
        with st.container(border=True):
            st.caption(
                f"Active chat model: `{response.debug.active_chat_model or st.session_state['active_chat_model']}`"
            )
            st.write(response.answer)
            if response.track_context_suggestions is not None and response.track_context is not None:
                st.markdown("#### Suggested Track Context Updates")
                st.caption("These are review-only suggestions from the latest answer. They are not saved until you apply them.")
                with st.container(border=True):
                    for label, value in suggestion_groups(response.track_context_suggestions):
                        st.markdown(f"**{label}**")
                        if isinstance(value, list):
                            for item in value:
                                st.write(f"- {item}")
                        else:
                            st.write(value)
                if st.button("Apply Suggested Updates", use_container_width=False):
                    try:
                        updated_context = query_service.track_context_service.apply_suggestions(
                            response.track_context.track_id,
                            response.track_context_suggestions,
                        )
                        st.session_state["last_query_response"] = QueryResponse(
                            answer_result=response.answer_result,
                            warnings=response.warnings,
                            linked_context_chunks=response.linked_context_chunks,
                            web_results=response.web_results,
                            saved_path=response.saved_path,
                            debug=response.debug,
                            domain_profile=response.domain_profile,
                            collaboration_workflow=response.collaboration_workflow,
                            workflow_input=response.workflow_input,
                            track_context=updated_context,
                            track_context_suggestions=None,
                        )
                        st.session_state["track_context_apply_success"] = "Suggested Track Context updates applied."
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            success_message = st.session_state.pop("track_context_apply_success", "")
            if success_message:
                st.success(success_message)

    detail_parent = chat_detail_mount if chat_workspace_enabled and chat_detail_mount is not None else st.container()
    with detail_parent:
        detail_container = st.expander("Latest Response Details", expanded=not chat_workspace_enabled)
    with detail_container:
        for warning in response.warnings:
            st.warning(warning)

        with st.expander("Answer Mode and Evidence Guide", expanded=False):
            st.write("Strict — Retrieved evidence only; says when evidence is missing.")
            st.write("Balanced — Evidence first, limited model reasoning.")
            st.write("Exploratory — Evidence plus broader synthesis and labeled inference.")
            st.write("Local = your Obsidian notes.")
            st.write("Web = external search results.")
            st.write("Inference = model synthesis beyond directly retrieved evidence.")

        if response.web_used and response.debug.retrieval_mode_requested == "auto":
            if response.debug.local_retrieval_weak and response.debug.primary_chunks:
                st.info(
                    "Web fallback was used because local note retrieval looked weak for this question."
                )
            elif not response.debug.primary_chunks:
                st.info(
                    "Web fallback was used because no strong local note context was available."
                )
        if response.debug.web_query_strategy.value == "local_guided" and response.web_used:
            st.info("External web search was narrowed using the strongest local note topics.")
        if response.debug.web_alignment_warning:
            st.info(response.debug.web_alignment_warning)
        if response.debug.web_attempts:
            st.caption(f"Web query used: {response.debug.web_query_used}")
        if response.debug.web_retry_used:
            st.info("A lighter retry web query was attempted after the first web-search attempt failed.")
        if not response.web_used and response.debug.web_failure_reason:
            st.warning(
                "No web sources were used. "
                + {
                    "provider_returned_no_results": "The provider returned no results for the attempted web query.",
                    "all_results_filtered_out": "Returned web results were discarded because they did not align with the local topic.",
                    "provider_error": "The web-search provider could not be reached successfully.",
                }.get(response.debug.web_failure_reason, "No usable web evidence was found.")
            )

        status_cols = st.columns(5)
        status_cols[0].metric("Answer mode", response.answer_mode_used.value)
        status_cols[1].metric("Retrieval mode", str(response.debug.retrieval_mode_used))
        status_cols[2].metric("Retrieval scope", response.debug.retrieval_scope_requested.value)
        status_cols[3].metric("Web used", "Yes" if response.web_used else "No")
        status_cols[4].metric("Inference used", "Yes" if response.inference_used else "No")

        trust_cols = st.columns(4)
        trust_cols[0].metric("Curated knowledge chunks", response.debug.curated_knowledge_chunks)
        trust_cols[1].metric("Imported knowledge chunks", response.debug.imported_knowledge_chunks)
        trust_cols[2].metric("Non-curated note chunks", response.debug.non_curated_note_chunks)
        trust_cols[3].metric("Generated draft chunks", response.debug.generated_or_imported_chunks)

        st.subheader("Latest Response Sources")
        with st.container():
            if response.curated_chunks:
                st.markdown("**Curated Knowledge**")
                for chunk in response.curated_chunks:
                    st.write(f"- {_source_line_from_chunk(chunk, label='[Local]')}")
            if response.imported_chunks:
                st.markdown("**Imported Knowledge**")
                for chunk in response.imported_chunks:
                    st.write(f"- {_source_line_from_chunk(chunk, label='[Import]')}")
            if response.non_curated_chunks:
                st.markdown("**Non-Curated Notes**")
                for chunk in response.non_curated_chunks:
                    st.write(f"- {_source_line_from_chunk(chunk, label='[Local]')}")
            if not response.curated_chunks and not response.imported_chunks and not response.non_curated_chunks:
                st.write("- No local note sources")
            if response.saved_sources:
                st.markdown("**Generated Draft Sources**")
                for source in response.saved_sources:
                    st.write(f"- {source}")
            if response.imported_sources:
                st.markdown("**Imported Sources**")
                for source in response.imported_sources:
                    st.write(f"- {source}")
            if response.web_sources:
                st.markdown("**Web Sources**")
                for source in response.web_sources:
                    st.write(f"- {source}")

        if response.web_used:
            st.info("External web results contributed to this answer.")

        if response.linked_context_chunks:
            with st.expander("Linked Note Context Used", expanded=False):
                linked_titles = sorted(
                    {
                        f"{chunk.metadata.get('note_title', 'Untitled')} ({chunk.metadata.get('source_path', 'unknown')})"
                        for chunk in response.linked_context_chunks
                    }
                )
                for linked_title in linked_titles:
                    st.write(f"- {linked_title}")

        save_disabled = response.has_saved or not (response.retrieved_chunks or response.web_results)
        save_help = "This answer has already been saved." if response.has_saved else None
        if st.button("Save To Vault", disabled=save_disabled, help=save_help):
            try:
                saved_response = query_service.save(
                    st.session_state.get("last_question", question.strip()),
                    response.answer_result,
                    title_override=st.session_state["save_title"].strip() or None,
                    existing_response=response,
                )
                st.session_state["last_query_response"] = saved_response
                st.success(f"Saved answer to {saved_response.saved_path}")
            except Exception as exc:
                st.error(str(exc))

        if st.session_state["debug_mode"]:
            _render_debug_section(response, st.session_state.get("last_question", ""))


def _render_research_response(
    question: str,
    response: ResearchResponse,
    research_service: ResearchService,
) -> None:
    if response.active_chat_model:
        st.caption(f"Active chat model: `{response.active_chat_model}`")
    for warning in response.warnings:
        st.warning(warning)

    with st.expander("Answer Mode and Evidence Guide", expanded=False):
        st.write("Research mode plans a small list of explicit subquestions, answers each one, then synthesizes a final response.")
        st.write("Strict — Retrieved evidence only; says when evidence is missing.")
        st.write("Balanced — Evidence first, limited model reasoning.")
        st.write("Exploratory — Evidence plus broader synthesis and labeled inference.")
        st.write("Local = your Obsidian notes.")
        st.write("Saved = previously saved answer notes, treated as secondary sources.")
        st.write("Web = external search results.")
        st.write("Inference = model synthesis beyond directly retrieved evidence.")

    status_cols = st.columns(5)
    status_cols[0].metric("Workflow", "Research")
    status_cols[1].metric("Subquestions", str(len(response.subquestions)))
    status_cols[2].metric(
        "Retrieval scope",
        response.steps[0].response.debug.retrieval_scope_requested.value if response.steps else RetrievalScope.KNOWLEDGE.value,
    )
    status_cols[3].metric("Web used", "Yes" if response.web_sources else "No")
    status_cols[4].metric("Inference used", "Yes" if response.inference_used else "No")

    st.subheader("Research Plan")
    for index, subquestion in enumerate(response.subquestions, start=1):
        st.write(f"{index}. {subquestion}")

    st.subheader("Step Findings")
    for index, step in enumerate(response.steps, start=1):
        with st.expander(f"Step {index}: {step.subquestion}", expanded=False):
            st.write(step.response.answer)
            if step.response.sources:
                st.markdown("**Sources**")
                for source in step.response.sources:
                    st.write(f"- {source}")
            if step.response.warnings:
                st.markdown("**Warnings**")
                for warning in step.response.warnings:
                    st.write(f"- {warning}")

    summary_col, sources_col = st.columns([3, 2])
    with summary_col:
        st.subheader("Final Research Answer")
        st.write(response.answer)
    with sources_col:
        st.subheader("Sources")
        curated_chunks = [
            chunk for chunk in response.retrieved_chunks if chunk.metadata.get("content_category") == "curated_knowledge"
        ]
        non_curated_chunks = [
            chunk for chunk in response.retrieved_chunks if chunk.metadata.get("content_category") == "non_curated_note"
        ]
        if curated_chunks:
            st.markdown("**Curated Knowledge**")
            for chunk in curated_chunks:
                st.write(f"- {_source_line_from_chunk(chunk, label='[Local]')}")
        imported_chunks = [
            chunk for chunk in response.retrieved_chunks if chunk.metadata.get("content_category") == "imported_knowledge"
        ]
        if imported_chunks:
            st.markdown("**Imported Knowledge**")
            for chunk in imported_chunks:
                st.write(f"- {_source_line_from_chunk(chunk, label='[Import]')}")
        if non_curated_chunks:
            st.markdown("**Non-Curated Notes**")
            for chunk in non_curated_chunks:
                st.write(f"- {_source_line_from_chunk(chunk, label='[Local]')}")
        if not curated_chunks and not imported_chunks and not non_curated_chunks:
            st.write("- No local note sources")
        if response.saved_sources:
            st.markdown("**Generated Draft Sources**")
            for source in response.saved_sources:
                st.write(f"- {source}")
        if response.imported_sources:
            st.markdown("**Imported Sources**")
            for source in response.imported_sources:
                st.write(f"- {source}")
        if response.web_sources:
            st.markdown("**Web Sources**")
            for source in response.web_sources:
                st.write(f"- {source}")

    save_disabled = response.has_saved or not response.sources
    save_help = "This research answer has already been saved." if response.has_saved else None
    if st.button("Save Research To Vault", disabled=save_disabled, help=save_help):
        try:
            saved_response = research_service.save(
                st.session_state.get("last_question", question.strip()),
                response.answer_result,
                title_override=st.session_state["save_title"].strip() or None,
                existing_response=response,
            )
            st.session_state["last_query_response"] = saved_response
            st.success(f"Saved research answer to {saved_response.saved_path}")
        except Exception as exc:
            st.error(str(exc))


def _active_yaml_track_context(query_service: QueryService):
    track_id = st.session_state.get("track_context_track_id", "").strip()
    if not st.session_state.get("use_track_context") or not track_id:
        return None
    return query_service.track_context_service.load_or_create(track_id)


def _render_current_track_summary(track_context) -> None:
    title, rows = current_track_summary(track_context)
    with st.container(border=True):
        st.markdown("### Current Track")
        if not rows:
            st.caption(title)
            return
        st.caption(title)
        left_col, right_col = st.columns(2)
        midpoint = (len(rows) + 1) // 2
        for label, value in rows[:midpoint]:
            left_col.write(f"**{label}:** {value}")
        for label, value in rows[midpoint:]:
            right_col.write(f"**{label}:** {value}")


def _render_ingest_tab(ingestion_service: IngestionService) -> None:
    st.caption(
        "Use ingestion to save external content into your vault as normal Markdown notes. "
        "This is separate from query-time web search, and imported notes are excluded from indexing by default."
    )

    webpage_col, youtube_col = st.columns(2)

    with webpage_col:
        st.subheader("Import a Webpage")
        st.caption("Saved into the configured webpage-imports folder for later review or promotion.")
        with st.form("ingest_webpage_form", clear_on_submit=False, enter_to_submit=False):
            st.text_input(
                "Webpage URL",
                placeholder="https://example.com/article",
                key="ingest_url",
            )
            st.text_input(
                "Optional note title",
                key="ingest_title",
                help="Override the saved note title and filename slug.",
            )
            st.checkbox(
                "Index immediately after save",
                help="Run the existing incremental index after creating the note.",
                key="ingest_index_now",
            )
            webpage_submit = st.form_submit_button("Ingest Webpage", type="primary", use_container_width=True)

        if webpage_submit:
            url = st.session_state["ingest_url"].strip()
            if not url:
                st.warning("Enter a webpage URL before starting ingestion.")
            else:
                try:
                    response = ingestion_service.ingest_webpage(
                        IngestionRequest(
                            source=url,
                            title_override=st.session_state["ingest_title"].strip() or None,
                            index_now=st.session_state["ingest_index_now"],
                        )
                    )
                    st.session_state["last_ingestion_response"] = response
                    st.success(f"Saved webpage note to {response.saved_path}")
                except Exception as exc:
                    st.session_state["last_ingestion_response"] = None
                    st.error(str(exc))

    with youtube_col:
        st.subheader("Import a YouTube Video")
        st.caption("Saved into the configured YouTube-imports folder for later review or promotion.")
        with st.form("ingest_youtube_form", clear_on_submit=False, enter_to_submit=False):
            st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                key="youtube_url",
            )
            st.text_input(
                "Optional video note title",
                key="youtube_title",
                help="Override the saved note title and filename slug.",
            )
            st.checkbox(
                "Index YouTube note immediately",
                help="Run the existing incremental index after creating the note.",
                key="youtube_index_now",
            )
            youtube_submit = st.form_submit_button("Ingest YouTube", type="primary", use_container_width=True)

        if youtube_submit:
            url = st.session_state["youtube_url"].strip()
            if not url:
                st.warning("Enter a YouTube URL before starting ingestion.")
            else:
                try:
                    response = ingestion_service.ingest_youtube(
                        IngestionRequest(
                            source=url,
                            title_override=st.session_state["youtube_title"].strip() or None,
                            index_now=st.session_state["youtube_index_now"],
                        )
                    )
                    st.session_state["last_ingestion_response"] = response
                    st.success(f"Saved YouTube note to {response.saved_path}")
                except Exception as exc:
                    st.session_state["last_ingestion_response"] = None
                    st.error(str(exc))

    response = st.session_state.get("last_ingestion_response")
    if response is None:
        return

    st.markdown("### Latest Ingestion")
    st.write(f"Title: `{response.title}`")
    st.write(f"Saved path: `{response.saved_path}`")
    st.write(f"Source type: `{response.source_type}`")
    st.write(f"Indexed now: `{'yes' if response.index_triggered else 'no'}`")
    for warning in response.warnings:
        st.warning(warning)


def _render_debug_section(response: QueryResponse, original_question: str) -> None:
    with st.expander("Debug Details", expanded=False):
        st.markdown("**Query Summary**")
        for label, value in debug_query_summary(original_question, response.debug.rewritten_query):
            st.write(f"{label}: `{value}`")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Initial candidates", len(response.debug.initial_candidates))
        metric_cols[1].metric("Primary chunks", len(response.debug.primary_chunks))
        metric_cols[2].metric("Final chunks", len(response.retrieved_chunks))
        metric_cols[3].metric("Reranking changed order", "Yes" if response.debug.reranking_changed else "No")

        st.markdown("**Retrieval / Search Details**")
        st.json(
            {
                "filters": {
                    "folder": response.debug.retrieval_filters.folder,
                    "path_contains": response.debug.retrieval_filters.path_contains,
                    "tag": response.debug.retrieval_filters.tag,
                },
                "options": {
                    "top_k": response.debug.retrieval_options.top_k,
                    "candidate_count": response.debug.retrieval_options.candidate_count,
                    "rerank": response.debug.retrieval_options.rerank,
                    "boost_tags": list(response.debug.retrieval_options.boost_tags),
                    "include_linked_notes": response.debug.retrieval_options.include_linked_notes,
                },
                "retrieval_scope_requested": response.debug.retrieval_scope_requested,
                "retrieval_mode_requested": response.debug.retrieval_mode_requested,
                "retrieval_mode_used": response.debug.retrieval_mode_used,
                "answer_mode_requested": response.debug.answer_mode_requested,
                "answer_mode_used": response.debug.answer_mode_used,
                "rewritten_query": response.debug.rewritten_query,
                "web_used": response.debug.web_used,
                "curated_knowledge_chunks": response.debug.curated_knowledge_chunks,
                "imported_knowledge_chunks": response.debug.imported_knowledge_chunks,
                "non_curated_note_chunks": response.debug.non_curated_note_chunks,
                "generated_or_imported_chunks": response.debug.generated_or_imported_chunks,
                "active_chat_model": response.debug.active_chat_model,
                "local_retrieval_weak": response.debug.local_retrieval_weak,
                "reranking_applied": response.debug.reranking_applied,
                "evidence_types_used": list(response.debug.evidence_types_used),
                "inference_used": response.debug.inference_used,
                "citation_labels": list(response.debug.citation_labels),
                "web_query_used": response.debug.web_query_used,
                "web_query_strategy": response.debug.web_query_strategy,
                "web_results_filtered_count": response.debug.web_results_filtered_count,
                "web_alignment_warning": response.debug.web_alignment_warning,
                "web_failure_reason": response.debug.web_failure_reason,
                "web_provider_returned_results": response.debug.web_provider_returned_results,
                "web_results_discarded_by_filter": response.debug.web_results_discarded_by_filter,
                "web_retry_used": response.debug.web_retry_used,
                "hallucination_guard_warnings": list(response.debug.hallucination_guard_warnings),
            }
        )
        st.markdown("**Web Search Attempts**")
        if not response.debug.web_attempts:
            st.caption("None")
        else:
            for index, attempt in enumerate(response.debug.web_attempts, start=1):
                st.json(
                    {
                        "attempt": index,
                        "query": attempt.query,
                        "strategy": attempt.strategy,
                        "retry_used": attempt.retry_used,
                        "provider_returned_results": attempt.provider_returned_results,
                        "provider_result_count": attempt.provider_result_count,
                        "usable_result_count": attempt.usable_result_count,
                        "filtered_count": attempt.filtered_count,
                        "results_discarded_by_filter": attempt.results_discarded_by_filter,
                        "failure_reason": attempt.failure_reason,
                        "outcome": attempt.outcome,
                    }
                )

        st.markdown("**Retrieved Chunks**")
        _render_chunk_list("Initial Retrieval Candidates", response.debug.initial_candidates)
        _render_chunk_list("Primary Selected Local Chunks", response.debug.primary_chunks)
        _render_chunk_list("Final Selected Chunks", response.retrieved_chunks)
        _render_web_results(response)


def _render_chunk_list(title: str, chunks: list) -> None:
    st.markdown(f"**{title}**")
    if not chunks:
        st.caption("None")
        return
    for index, chunk in enumerate(chunks, start=1):
        st.markdown(f"**Chunk {index}**")
        st.json(
            {
                "source_path": chunk.metadata.get("source_path"),
                "note_title": chunk.metadata.get("note_title"),
                "heading_context": chunk.metadata.get("heading_context"),
                "content_scope": chunk.metadata.get("content_scope"),
                "content_category": chunk.metadata.get("content_category"),
                "source_type": chunk.metadata.get("source_type"),
                "distance_or_score": chunk.distance_or_score,
                "linked_context": chunk.metadata.get("linked_context", False),
                "tags_serialized": chunk.metadata.get("tags_serialized", ""),
            }
        )
        st.code(chunk.text)


def _render_web_results(response: QueryResponse) -> None:
    st.markdown("**Web Results**")
    if not response.web_results:
        st.caption("None")
        return
    for index, result in enumerate(response.web_results, start=1):
        st.markdown(f"**Web Result {index}**")
        st.json(
            {
                "title": result.title,
                "url": result.url,
                "source_type": result.source_type,
            }
        )
        st.write(result.snippet)


def _source_line_from_chunk(chunk, *, label: str) -> str:
    return (
        f"{label} {chunk.metadata.get('note_title', 'Untitled')} "
        f"({chunk.metadata.get('source_path', 'unknown')})"
    )


def _workflow_label(value: str) -> str:
    return {
        CollaborationWorkflow.GENERAL_ASK.value: "General Ask",
        CollaborationWorkflow.GENRE_FIT_REVIEW.value: "Genre Fit Review",
        CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE.value: "Track Concept Critique",
        CollaborationWorkflow.ARRANGEMENT_PLANNER.value: "Arrangement Planner",
        CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM.value: "Sound Design Brainstorm",
        CollaborationWorkflow.RESEARCH_SESSION.value: "Research Session",
    }.get(value, value.replace("_", " ").title())


def _workflow_help_text(workflow: CollaborationWorkflow) -> str:
    return {
        CollaborationWorkflow.GENERAL_ASK: "Quick collaboration for track ideas, references, production decisions, or note-grounded questions.",
        CollaborationWorkflow.GENRE_FIT_REVIEW: "Assess whether an idea fits a target style and identify the strongest genre cues, mismatches, and refinements.",
        CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE: "Pressure-test a track concept, then sharpen the arrangement, energy, and sound-design direction.",
        CollaborationWorkflow.ARRANGEMENT_PLANNER: "Turn a rough idea into a section-by-section production plan with pacing and transition guidance.",
        CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM: "Explore synth, drum, bass, texture, and FX directions with practical production considerations.",
        CollaborationWorkflow.RESEARCH_SESSION: "Deeper multi-step workflow. It may break the topic into subquestions, gather broader evidence, and synthesize a final research answer.",
    }[workflow]


def _render_index_tab(index_service: IndexService, status: IndexResponse | None) -> None:
    st.subheader("Index Status")
    if status is None:
        st.warning("Unable to read index status right now.")
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Stored chunks", status.total_chunks_stored)
        metric_cols[1].metric("Index compatible", "Yes" if status.index_compatible else "No")
        metric_cols[2].metric("Ready", "Yes" if status.ready else "No")
        metric_cols[3].metric("Ollama", "Reachable" if status.ollama_reachable else "Unavailable")

        if status.ready:
            st.success("The assistant is ready to answer questions.")
        elif status.total_chunks_stored == 0:
            st.info("No indexed chunks found yet. Build the index to get started.")
        elif not status.index_compatible:
            st.warning("The existing index needs a rebuild before it should be used.")

        if status.ollama_status_message:
            if status.ollama_reachable:
                st.caption(status.ollama_status_message)
            else:
                st.warning(status.ollama_status_message)

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("Build Index", type="primary"):
        try:
            response = index_service.index(reset_store=False)
            _show_index_result("Index complete", response)
        except Exception as exc:
            st.error(str(exc))

    if action_col2.button("Rebuild Index"):
        try:
            response = index_service.index(reset_store=True)
            _show_index_result("Rebuild complete", response)
        except Exception as exc:
            st.error(str(exc))


def _show_index_result(title: str, response: IndexResponse) -> None:
    st.success(
        f"{title}. Notes: {response.notes_loaded}, chunks created: {response.chunks_created}, "
        f"chunks indexed: {response.chunks_indexed}, stored chunks: {response.total_chunks_stored}"
    )
    if response.up_to_date:
        st.info("The index was already up to date.")
    for warning in response.warnings:
        st.warning(warning)


def _render_settings_tab(config: AppConfig, status: IndexResponse | None, status_error: str | None) -> None:
    available_chat_models, _ = list_available_chat_models(config)
    active_chat_model = _resolve_preferred_chat_model_name(
        st.session_state.get("active_chat_model", _DEFAULT_ACTIVE_CHAT_MODEL),
        available_chat_models,
    )
    st.subheader("Active Models")
    model_cols = st.columns(2)
    model_cols[0].write(f"Chat model: `{active_chat_model}`")
    model_cols[1].write(f"Embedding model: `{config.ollama_embedding_model}`")

    st.subheader("Paths and Status")
    if status is None:
        st.warning("Status is currently unavailable.")
        if status_error:
            st.caption(status_error)
    else:
        st.write(f"Vault path: `{status.vault_path}`")
        st.write(f"Draft answers path: `{config.draft_answers_path}`")
        st.write(f"Research sessions path: `{config.research_sessions_path}`")
        st.write(f"Curated knowledge folder: `{config.curated_knowledge_path}`")
        st.write(f"Webpage imports folder: `{config.webpage_ingestion_path}`")
        st.write(f"YouTube imports folder: `{config.youtube_ingestion_path}`")
        st.write(f"Index schema version: `{status.index_version or 'not set'}`")
        st.write(f"Stored chunks: `{status.total_chunks_stored}`")
        st.write(f"Index compatible: `{'yes' if status.index_compatible else 'no'}`")
        st.write(f"App ready: `{'yes' if status.ready else 'no'}`")
        st.caption(
            "By default, indexing excludes draft answers, research sessions, webpage imports, and YouTube imports. "
            "Those folders can be enabled explicitly in config if you want them indexed."
        )

    st.subheader("Advanced / Debug")
    st.session_state["debug_mode"] = st.checkbox(
        "Show debug details in Ask results",
        value=st.session_state["debug_mode"],
    )
    st.caption(
        "Query filters and retrieval controls live in the sidebar so they stay close to the Ask workflow."
    )
    st.write(f"Current retrieval scope: `{st.session_state['retrieval_scope']}`")
    st.write(f"Current retrieval mode: `{st.session_state['retrieval_mode']}`")
    st.write(f"Current answer mode: `{st.session_state['answer_mode']}`")
    st.write(f"Current collaboration workflow: `{st.session_state['collaboration_workflow']}`")


def _current_filters() -> RetrievalFilters:
    return RetrievalFilters(
        tag=st.session_state["tag_filter"].strip().lstrip("#").lower() or None,
        folder=st.session_state["folder_filter"].strip().strip("/") or None,
        path_contains=st.session_state["path_filter"].strip().lower() or None,
    )


def _current_options() -> RetrievalOptions:
    return RetrievalOptions(
        top_k=st.session_state["top_k"],
        rerank=st.session_state["enable_reranking"],
        include_linked_notes=st.session_state["include_linked"],
    )


def _current_workflow_input() -> WorkflowInput:
    return WorkflowInput(
        genre=st.session_state["workflow_genre"],
        bpm=st.session_state["workflow_bpm"],
        references=st.session_state["workflow_references"],
        mood=st.session_state["workflow_mood"],
        arrangement_notes=st.session_state["workflow_arrangement_notes"],
        instrumentation=st.session_state["workflow_instrumentation"],
        sound_palette=st.session_state["workflow_sound_palette"],
        energy_goal=st.session_state["workflow_energy_goal"],
        track_length=st.session_state["workflow_track_length"],
        role_of_key_elements=st.session_state["workflow_role_of_key_elements"],
        track_context_path=st.session_state["workflow_track_context_path"],
    )


def _apply_legacy_track_selection(selected_track: str, tracks: list[dict[str, str]]) -> None:
    current_path = st.session_state.get("workflow_track_context_path", "").strip()
    resolved_path = selected_track_path(selected_track, tracks)
    if resolved_path is None:
        if current_path in {track["path"] for track in tracks}:
            st.session_state["workflow_track_context_path"] = ""
        return
    if current_path != resolved_path:
        st.session_state["workflow_track_context_path"] = resolved_path


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _dedupe_chat_model_options(models: list[str]) -> list[str]:
    canonical_models = []
    seen: set[str] = set()
    for model in models:
        canonical = _resolve_preferred_chat_model_name(model, models)
        if canonical in seen:
            continue
        seen.add(canonical)
        canonical_models.append(canonical)
    return canonical_models


def _resolve_preferred_chat_model_name(model: str, available_models: list[str]) -> str:
    normalized_model = model.strip()
    if not normalized_model:
        return _DEFAULT_ACTIVE_CHAT_MODEL
    if normalized_model in available_models:
        return normalized_model
    if f"{normalized_model}:latest" in available_models:
        return f"{normalized_model}:latest"
    for candidate in available_models:
        if candidate.startswith(f"{normalized_model}-") or candidate.startswith(f"{normalized_model}:"):
            return candidate
    return normalized_model


def _render_chat_history() -> None:
    for index, message in enumerate(st.session_state.get("chat_messages", [])):
        with st.chat_message(message.role):
            st.write(message.content)
            st.caption(f"{message.role.title()} message {index + 1} • {message.created_at}")


def _render_chat_debug_panel(workflow_value: str) -> None:
    """Temporary Ask-tab debug output for chat-state verification."""
    with st.expander("Temporary Chat Debug", expanded=False):
        chat_messages = st.session_state.get("chat_messages", [])
        last_message = chat_messages[-1] if chat_messages else None
        last_preview = (
            f"{last_message.role}: {last_message.content[:120]}"
            if last_message is not None
            else "No messages yet."
        )
        st.caption("Temporary debug block for chat-state verification. Safe to remove after the display issue is confirmed fixed.")
        st.write(f"Workflow: `{workflow_value}`")
        st.write(f"Chat workspace enabled: `{workflow_value in _CHAT_TASK_WORKFLOWS}`")
        st.write(f"Chat message count: `{len(chat_messages)}`")
        st.write(f"Last message preview: `{last_preview}`")
        st.write(f"last_query_response exists: `{st.session_state.get('last_query_response') is not None}`")


def _render_task_panel() -> None:
    with st.expander("Session Tasks", expanded=False):
        st.caption("Tasks are session-only and are used as internal execution context for critique, arrangement, and sound-design workflows.")
        with st.form("add_task_form", clear_on_submit=False, enter_to_submit=False):
            st.text_input("Task", key="new_task_text", placeholder="Add a focused production task")
            st.text_input("Notes (optional)", key="new_task_notes", placeholder="Optional detail")
            add_task = st.form_submit_button("Add Task", use_container_width=True)

        if add_task and st.session_state["new_task_text"].strip():
            tasks = list(st.session_state["session_tasks"])
            tasks.append(
                SessionTask(
                    id=uuid4().hex,
                    text=st.session_state["new_task_text"].strip(),
                    status="open",
                    source="user",
                    created_at=current_timestamp(),
                    notes=st.session_state["new_task_notes"].strip(),
                )
            )
            st.session_state["session_tasks"] = tasks
            st.session_state["new_task_text"] = ""
            st.session_state["new_task_notes"] = ""
            st.rerun()

        open_tasks = [task for task in st.session_state.get("session_tasks", []) if task.status == "open"]
        completed_tasks = [task for task in st.session_state.get("session_tasks", []) if task.status == "completed"]

        st.markdown("**Open Tasks**")
        if not open_tasks:
            st.caption("No open tasks in this session.")
        for task in open_tasks:
            _render_task_actions(task)

        st.markdown("**Completed Tasks**")
        if not completed_tasks:
            st.caption("No completed tasks yet.")
        for task in completed_tasks:
            _render_task_actions(task)


def _render_task_actions(task: SessionTask) -> None:
    cols = st.columns([4, 1, 1])
    note_suffix = f" ({task.notes})" if task.notes else ""
    cols[0].write(f"{'[ ]' if task.status == 'open' else '[x]'} {task.text}{note_suffix}")
    toggle_label = "Done" if task.status == "open" else "Reopen"
    if cols[1].button(toggle_label, key=f"task-toggle-{task.id}", use_container_width=True):
        updated_tasks: list[SessionTask] = []
        for existing_task in st.session_state["session_tasks"]:
            if existing_task.id == task.id:
                updated_tasks.append(
                    replace(
                        existing_task,
                        status="completed" if existing_task.status == "open" else "open",
                    )
                )
            else:
                updated_tasks.append(existing_task)
        st.session_state["session_tasks"] = updated_tasks
        st.rerun()
    if cols[2].button("Delete", key=f"task-delete-{task.id}", use_container_width=True):
        st.session_state["session_tasks"] = [
            existing_task for existing_task in st.session_state["session_tasks"] if existing_task.id != task.id
        ]
        st.rerun()


def _recent_conversation_for_prompt(
    chat_messages: list[ChatMessage],
    workflow_value: str,
) -> list[ChatMessage]:
    if workflow_value not in _CHAT_TASK_WORKFLOWS:
        return []
    return chat_messages[-8:]


def _tasks_for_prompt(
    session_tasks: list[SessionTask],
    workflow_value: str,
) -> list[SessionTask]:
    if workflow_value not in _CHAT_TASK_WORKFLOWS:
        return []
    open_tasks = [task for task in session_tasks if task.status == "open"]
    completed_tasks = [task for task in session_tasks if task.status == "completed"]
    return open_tasks + completed_tasks


def _submit_question_from_input() -> None:
    st.session_state["submitted_message"] = st.session_state.get("question_input", "").strip()
    st.session_state["question_input"] = ""


def _init_session_state(config: AppConfig) -> None:
    defaults = {
        "question_input": "",
        "submitted_message": "",
        "folder_filter": "",
        "path_filter": "",
        "tag_filter": "",
        "save_title": "",
        "active_chat_model": _DEFAULT_ACTIVE_CHAT_MODEL,
        "active_chat_model_select": _DEFAULT_ACTIVE_CHAT_MODEL,
        "active_chat_model_input": _DEFAULT_ACTIVE_CHAT_MODEL,
        "top_k": config.top_k_results,
        "enable_reranking": config.enable_reranking,
        "include_linked": config.enable_linked_note_expansion,
        "retrieval_scope": RetrievalScope.KNOWLEDGE.value,
        "auto_save": config.auto_save_answer,
        "retrieval_mode": RetrievalMode.AUTO.value,
        "answer_mode": AnswerMode.BALANCED.value,
        "workflow_mode": WorkflowMode.DIRECT.value,
        "collaboration_workflow": CollaborationWorkflow.GENERAL_ASK.value,
        "workflow_genre": "",
        "workflow_bpm": "",
        "workflow_references": "",
        "workflow_mood": "",
        "workflow_arrangement_notes": "",
        "workflow_instrumentation": "",
        "workflow_sound_palette": "",
        "workflow_energy_goal": "",
        "workflow_track_length": "",
        "workflow_role_of_key_elements": "",
        "workflow_track_context_path": "",
        "workflow_track_selector": "None",
        "track_context_track_id": "",
        "use_track_context": True,
        "track_context_track_name": "",
        "track_context_genre": "",
        "track_context_bpm": "",
        "track_context_key": "",
        "track_context_workflow_mode": "general",
        "track_context_current_stage": "",
        "track_context_current_section": "",
        "track_context_vibe": "",
        "track_context_reference_tracks": "",
        "track_context_known_issues": "",
        "track_context_goals": "",
        "track_context_notes": "",
        "chat_messages": [],
        "session_tasks": [],
        "new_task_text": "",
        "new_task_notes": "",
        "max_subquestions": 3,
        "debug_mode": False,
        "last_query_response": None,
        "last_question": "",
        "ingest_url": "",
        "ingest_title": "",
        "ingest_index_now": config.auto_index_after_ingestion,
        "youtube_url": "",
        "youtube_title": "",
        "youtube_index_now": config.auto_index_after_ingestion,
        "last_ingestion_response": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _config_from_session(config: AppConfig) -> AppConfig:
    return replace(
        config,
        top_k_results=int(st.session_state.get("top_k", config.top_k_results)),
        enable_reranking=bool(st.session_state.get("enable_reranking", config.enable_reranking)),
        enable_linked_note_expansion=bool(
            st.session_state.get("include_linked", config.enable_linked_note_expansion)
        ),
        auto_save_answer=bool(st.session_state.get("auto_save", config.auto_save_answer)),
    )


_CHAT_TASK_WORKFLOWS = {
    CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE.value,
    CollaborationWorkflow.ARRANGEMENT_PLANNER.value,
    CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM.value,
}

_TRACK_CONTEXT_WORKFLOW_MODES = [
    "general",
    "track_critique",
    "composition",
    "arrangement",
    "sound_design",
    "critique",
    "mixing",
    "research",
]

_TRACK_CONTEXT_CURRENT_STAGES = [
    "idea",
    "sketch",
    "writing",
    "arrangement",
    "sound_design",
    "production",
    "mixing",
    "mastering",
    "finalizing",
]

_DEFAULT_ACTIVE_CHAT_MODEL = "deepseek"


if __name__ == "__main__":
    main()
