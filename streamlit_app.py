"""Local Streamlit UI for Obsidian Track Collaborator."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import streamlit as st

from config import AppConfig, load_config
from model_provider import (
    configured_chat_model,
    configured_embedding_model,
    effective_chat_provider,
    list_available_chat_models,
)
from services.ingestion_service import IngestionService
from services.index_service import IndexService
from services.import_genre_service import GENERIC_IMPORT_GENRE, ImportGenreService
from services.knowledge_category_service import (
    GENERIC_KNOWLEDGE_CATEGORY_LABEL,
    KnowledgeCategoryService,
)
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
    TrackContext,
    WorkflowInput,
    WorkflowMode,
)
from services.query_service import QueryService
from services.research_service import ResearchService
from services.track_selector_service import (
    TrackSelectorService,
    selected_track_index,
    selected_track_path,
)
from services.track_context_update_review import proposal_groups
from services.ui_session_helpers import (
    DEV_MODE_PRESET_MANUAL,
    critique_support_summary,
    current_track_summary,
    dev_mode_preset_options,
    debug_query_summary,
    resolve_dev_mode_preset,
    synced_dev_mode_preset_selection,
    synced_chat_provider_selection,
    track_context_status,
)
from utils import RetrievalFilters, RetrievalOptions, current_timestamp


st.set_page_config(page_title="Obsidian Track Collaborator", layout="wide")


def main() -> None:
    """Render the Streamlit app."""
    st.title("Obsidian Track Collaborator")
    st.caption("Local-first collaboration for genre fit, track critique, arrangement planning, sound design, and research.")

    try:
        base_config = load_config()
    except Exception as exc:
        st.error(str(exc))
        return

    _init_session_state(base_config)
    _sync_dev_mode_preset_with_session()
    _sync_active_chat_provider_with_session(base_config)
    _sync_active_chat_model_with_available_models(base_config)
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
        st.caption(
            "Persistent memory for the current in-progress track. This YAML Track Context is the primary track-state system."
        )
        st.text_input(
            "Track ID",
            key="track_context_track_id",
            help="Stable internal ID for your in-progress track. Use the same ID to reopen the same persistent track memory later.",
        )
        st.checkbox(
            "Use Track Context",
            key="use_track_context",
            help="Enable persistent YAML track memory for asks and research workflows.",
        )
        typed_track_id = st.session_state.get("track_context_track_id", "").strip()
        active_track_id = st.session_state.get("active_track_context_id", "").strip()
        load_col, clear_col = st.columns(2)
        load_clicked = load_col.button(
            "Load Track Context",
            use_container_width=True,
            disabled=not typed_track_id,
            help="Load an existing persistent track memory or initialize a new one for this Track ID.",
        )
        clear_active_clicked = clear_col.button(
            "Clear Active Track",
            use_container_width=True,
            disabled=not active_track_id,
            help="Clear the active Track Context for this session without deleting the saved YAML file.",
        )
        if load_clicked and typed_track_id:
            context_exists = query_service.track_context_service.canonical_exists(typed_track_id)
            loaded_context = query_service.track_context_service.load_or_create_canonical_track_context(
                typed_track_id
            )
            st.session_state["active_track_context_id"] = typed_track_id
            st.session_state["active_track_context_loaded_existing"] = context_exists
            st.session_state["current_track_context"] = loaded_context
            st.session_state["session_tasks"] = query_service.track_task_service.load_session_tasks(typed_track_id)
            st.session_state["session_tasks_track_id"] = typed_track_id
            st.session_state["active_section_focus"] = ""
            st.session_state["track_context_editor_synced_track_id"] = ""
            st.rerun()
        if clear_active_clicked:
            st.session_state["active_track_context_id"] = ""
            st.session_state["active_track_context_loaded_existing"] = False
            st.session_state["current_track_context"] = None
            st.session_state["session_tasks"] = []
            st.session_state["session_tasks_track_id"] = ""
            st.session_state["active_section_focus"] = ""
            st.session_state["track_context_editor_synced_track_id"] = ""
            st.rerun()

        sidebar_context = None
        if st.session_state.get("use_track_context") and active_track_id:
            sidebar_context = st.session_state.get("current_track_context")
            if sidebar_context is None or getattr(sidebar_context, "track_id", "") != active_track_id:
                sidebar_context = query_service.track_context_service.load_canonical_track_context(active_track_id)
                st.session_state["current_track_context"] = sidebar_context
            if st.session_state.get("session_tasks_track_id", "") != active_track_id:
                st.session_state["session_tasks"] = query_service.track_task_service.load_session_tasks(active_track_id)
                st.session_state["session_tasks_track_id"] = active_track_id
            _sync_track_context_editor_state(sidebar_context)
            status_title, status_caption = track_context_status(
                use_track_context=True,
                entered_track_id=typed_track_id,
                active_track_id=active_track_id,
                existed_before_load=st.session_state.get("active_track_context_loaded_existing", False),
                track_context=sidebar_context,
            )
            st.caption(status_title)
            st.caption(status_caption)
            if sidebar_context.sections:
                with st.expander("Track Sections", expanded=False):
                    for section_key, section in sidebar_context.sections.items():
                        parts = []
                        if section.role:
                            parts.append(f"role: {section.role}")
                        if section.energy_level:
                            parts.append(f"energy: {section.energy_level}")
                        if section.bars:
                            parts.append(f"bars: {section.bars}")
                        summary = " | ".join(parts) if parts else "No extra section details yet."
                        st.write(f"**{section.name or section_key}**")
                        st.caption(summary)
            with st.expander("Edit Persistent Track Context", expanded=False):
                with st.form("track_context_editor", clear_on_submit=False, enter_to_submit=False):
                    st.caption("Core Track Identity")
                    core_col_left, core_col_right = st.columns(2)
                    core_col_left.text_input(
                        "Title",
                        key="track_context_track_name",
                    )
                    core_col_right.text_input("Genre", key="track_context_genre")
                    detail_col_left, detail_col_right = st.columns(2)
                    detail_col_left.text_input(
                        "BPM",
                        key="track_context_bpm",
                    )
                    detail_col_right.text_input("Key", key="track_context_key")
                    st.caption("Current Production State")
                    focus_col_left, focus_col_right = st.columns(2)
                    stage_options = [""] + _TRACK_CONTEXT_CURRENT_STAGES
                    focus_col_left.selectbox(
                        "Stage",
                        options=stage_options,
                        key="track_context_current_stage",
                    )
                    focus_col_right.text_input(
                        "Current Problem",
                        key="track_context_current_problem",
                    )
                    st.text_input(
                        "Vibe (comma separated)",
                        key="track_context_vibe",
                    )
                    st.caption("References, Issues, and Goals")
                    st.text_area(
                        "References (one per line)",
                        key="track_context_reference_tracks",
                        height=70,
                    )
                    st.text_area(
                        "Known Issues (one per line)",
                        key="track_context_known_issues",
                        height=70,
                    )
                    st.text_area(
                        "Goals (one per line)",
                        key="track_context_goals",
                        height=70,
                    )
                    saved = st.form_submit_button("Save Track Context", use_container_width=True)
                if saved:
                    updated_context = query_service.track_context_service.update_fields(
                        active_track_id,
                        {
                            "track_name": st.session_state["track_context_track_name"],
                            "genre": st.session_state["track_context_genre"],
                            "bpm": st.session_state["track_context_bpm"],
                            "key": st.session_state["track_context_key"],
                            "current_stage": st.session_state["track_context_current_stage"],
                            "current_problem": st.session_state["track_context_current_problem"],
                            "vibe": _split_csv(st.session_state["track_context_vibe"]),
                            "reference_tracks": _split_lines(
                                st.session_state["track_context_reference_tracks"]
                            ),
                            "known_issues": _split_lines(
                                st.session_state["track_context_known_issues"]
                            ),
                            "goals": _split_lines(st.session_state["track_context_goals"]),
                        },
                    )
                    st.session_state["current_track_context"] = updated_context
                    st.session_state["active_track_context_loaded_existing"] = True
                    st.success("Track context saved.")
        elif st.session_state.get("use_track_context"):
            status_title, status_caption = track_context_status(
                use_track_context=True,
                entered_track_id=typed_track_id,
                active_track_id=active_track_id,
                existed_before_load=st.session_state.get("active_track_context_loaded_existing", False),
                track_context=sidebar_context,
            )
            st.caption(status_title)
            st.caption(status_caption)
        else:
            status_title, status_caption = track_context_status(
                use_track_context=False,
                entered_track_id=typed_track_id,
                active_track_id=active_track_id,
                existed_before_load=st.session_state.get("active_track_context_loaded_existing", False),
                track_context=sidebar_context,
            )
            st.caption(status_title)
            st.caption(status_caption)

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
        st.caption(f"Chat provider/model: {config.chat_provider} / {configured_chat_model(config)}")
        st.caption(f"Embedding provider/model: {config.embedding_provider} / {configured_embedding_model(config)}")


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
            "workflow_track_selector_applied_path",
            "track_context_track_id",
            "track_context_track_name",
            "track_context_genre",
            "track_context_bpm",
            "track_context_key",
            "track_context_current_problem",
            "track_context_vibe",
            "track_context_reference_tracks",
            "track_context_known_issues",
            "track_context_goals",
            "new_task_text",
            "new_task_notes",
        ):
            st.session_state[key] = ""
        st.session_state["active_track_context_id"] = ""
        st.session_state["active_track_context_loaded_existing"] = False
        st.session_state["current_track_context"] = None
        st.session_state["active_section_focus"] = ""
        st.session_state["track_context_editor_synced_track_id"] = ""
        st.session_state["collaboration_workflow"] = CollaborationWorkflow.GENERAL_ASK.value
        st.session_state["workflow_mode"] = WorkflowMode.DIRECT.value
        st.session_state["use_track_context"] = True
        st.session_state["track_context_current_stage"] = ""
        st.session_state["max_subquestions"] = 3
        st.session_state["chat_messages"] = []
        st.session_state["session_tasks"] = []
        st.session_state["last_query_response"] = None
        st.session_state["last_question"] = ""
        st.session_state["dev_mode_preset_select"] = DEV_MODE_PRESET_MANUAL
        st.session_state["dev_mode_preset"] = ""
        st.session_state["last_synced_dev_mode_preset"] = ""
        st.session_state["active_chat_provider_select"] = f"Use configured default ({config.chat_provider})"
        st.session_state["chat_provider_override"] = ""
        st.session_state["last_synced_chat_provider_override"] = ""
        st.session_state["active_chat_model_select"] = f"Use configured default ({configured_chat_model(config)})"
        st.session_state["active_chat_model_input"] = ""
        st.session_state["chat_model_override"] = ""
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
    active_chat_provider = _effective_chat_provider(config)
    available_chat_models, chat_model_discovery_error = list_available_chat_models(
        config,
        provider_override=_session_chat_provider_override() or None,
    )
    active_track_context = _active_yaml_track_context(query_service)

    main_col, control_col = st.columns([3, 1.4], gap="large")
    with control_col:
        st.markdown("### Workspace Controls")
        st.caption("Workflow, context, reset, and tasks stay close by without interrupting the conversation flow.")
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
            st.caption("Legacy Markdown Workflow Context")
            st.caption(
                "Optional path-based compatibility input for older `track_context.md` files. "
                "Use YAML Track Context as the primary track-state system."
            )
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
            _apply_legacy_track_selection(config, selected_track, legacy_tracks)
            if selected_track != "None":
                resolved_legacy_path = selected_track_path(selected_track, legacy_tracks)
                if resolved_legacy_path:
                    st.caption(f"Loaded legacy markdown context from `{resolved_legacy_path}`.")
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
        preset_active = bool(_session_dev_mode_preset())
        st.markdown("#### Dev Mode Preset")
        st.selectbox(
            "Session preset",
            options=dev_mode_preset_options(),
            key="dev_mode_preset_select",
            help="Quickly switch chat provider and model together for this session. Embeddings remain Ollama-only.",
        )
        if preset_active:
            st.caption(f"Preset mode is active: `{_session_dev_mode_preset()}`")
        else:
            st.caption("Preset mode is off. Provider and model controls below are in manual mode.")

        st.markdown("#### Chat Provider")
        st.caption(f"Configured default: `{config.chat_provider}`")
        provider_options = [
            f"Use configured default ({config.chat_provider})",
            *_ordered_chat_provider_options(config.chat_provider),
        ]
        st.selectbox(
            "Session chat provider",
            options=provider_options,
            key="active_chat_provider_select",
            help="Optional session-only override for chat generation. Embeddings remain Ollama-based.",
            disabled=preset_active,
        )

        st.markdown("#### Chat Model")
        st.caption(f"Active provider: `{active_chat_provider}`")
        st.caption(
            f"Configured default for active provider: `{configured_chat_model(config, provider_override=_session_chat_provider_override() or None) or '(not configured)'}`"
        )
        if active_chat_provider == "ollama" and available_chat_models:
            current_model = _resolve_preferred_chat_model_name(
                _effective_chat_model(config),
                available_chat_models,
            )
            model_options = [
                f"Use configured default ({configured_chat_model(config, provider_override=_session_chat_provider_override() or None)})"
            ] + _dedupe_chat_model_options(
                available_chat_models
            )
            if current_model not in model_options[1:]:
                model_options.append(current_model)
            st.selectbox(
                "Session chat model",
                options=model_options,
                key="active_chat_model_select",
                help="Use the configured default model or choose a session-only override.",
                disabled=preset_active,
            )
        else:
            st.text_input(
                "Session chat model override",
                key="active_chat_model_input",
                placeholder=configured_chat_model(
                    config,
                    provider_override=_session_chat_provider_override() or None,
                )
                or "Enter a chat model name",
                help="Optional session-only override. Leave blank to use the configured default model.",
                disabled=preset_active,
            )
        workspace_updated = st.button("Update Workspace", use_container_width=True)
        reset_chat_overrides = st.button("Reset All Chat Overrides", use_container_width=True)

        if workspace_updated:
            selected_preset = st.session_state.get("dev_mode_preset_select", DEV_MODE_PRESET_MANUAL).strip()
            st.session_state["dev_mode_preset"] = "" if selected_preset == DEV_MODE_PRESET_MANUAL else selected_preset
            st.session_state["last_synced_dev_mode_preset"] = st.session_state["dev_mode_preset"]
            if _session_dev_mode_preset():
                resolved_preset = resolve_dev_mode_preset(
                    _session_dev_mode_preset(),
                    configured_ollama_model=configured_chat_model(config, provider_override="ollama"),
                    available_ollama_models=_dedupe_chat_model_options(
                        list_available_chat_models(config, provider_override="ollama")[0]
                    ),
                )
                st.session_state["chat_provider_override"] = resolved_preset[0] if resolved_preset else ""
                st.session_state["chat_model_override"] = resolved_preset[1] if resolved_preset else ""
                st.rerun()
            else:
                selected_provider_option = st.session_state.get("active_chat_provider_select", "").strip()
                default_provider_option = f"Use configured default ({config.chat_provider})"
                previous_provider = _session_chat_provider_override()
                if not selected_provider_option or selected_provider_option == default_provider_option:
                    st.session_state["chat_provider_override"] = ""
                else:
                    st.session_state["chat_provider_override"] = selected_provider_option

                provider_changed = previous_provider != _session_chat_provider_override()
                if provider_changed:
                    st.session_state["chat_model_override"] = ""
                    st.rerun()

                if _effective_chat_provider(config) == "ollama" and available_chat_models:
                    selected_option = st.session_state.get("active_chat_model_select", "").strip()
                    default_option = (
                        f"Use configured default ({configured_chat_model(config, provider_override=_session_chat_provider_override() or None)})"
                    )
                    if not selected_option or selected_option == default_option:
                        st.session_state["chat_model_override"] = ""
                    else:
                        st.session_state["chat_model_override"] = _resolve_preferred_chat_model_name(
                            selected_option,
                            available_chat_models,
                        )
                else:
                    st.session_state["chat_model_override"] = st.session_state.get("active_chat_model_input", "").strip()

        if reset_chat_overrides:
            st.session_state["dev_mode_preset"] = ""
            st.session_state["last_synced_dev_mode_preset"] = ""
            st.session_state["chat_provider_override"] = ""
            st.session_state["chat_model_override"] = ""
            st.session_state["last_synced_chat_provider_override"] = ""
            st.rerun()

        if st.session_state["retrieval_scope"] == RetrievalScope.KNOWLEDGE.value:
            st.caption(
                "Knowledge searches curated Knowledge folders plus indexed imported reference material such as webpages and YouTube transcripts."
            )
        else:
            st.caption(
                "Extended searches Knowledge, plus indexed working notes and Saved Outputs."
            )
        if _session_dev_mode_preset():
            st.caption(f"Active preset: `{_session_dev_mode_preset()}`")
        else:
            st.caption("Active preset: `manual`")
        if _session_chat_provider_override().strip():
            st.caption(
                f"Using session override provider: `{_effective_chat_provider(config)}`. Embeddings still use `{config.embedding_provider}`."
            )
        else:
            st.caption(f"Using configured default chat provider: `{config.chat_provider}`")
        if _session_chat_model_override().strip():
            st.caption(f"Using session override model: `{_effective_chat_model(config)}`")
        else:
            st.caption(
                f"Using configured default model: `{configured_chat_model(config, provider_override=_session_chat_provider_override() or None) or _DEFAULT_ACTIVE_CHAT_MODEL}`"
            )
        if chat_model_discovery_error:
            st.caption(f"Model discovery unavailable: {chat_model_discovery_error}")

        if st.button("Reset Session", use_container_width=True):
            clear_clicked = True
        st.caption("Reset Session clears the current chat, tasks, and composer/workflow context for this session.")

        if active_track_context is not None:
            active_track_name = active_track_context.track_name or active_track_context.track_id
            st.caption(f"Active track memory: `{active_track_name}` (`{active_track_context.track_id}`).")
            if st.session_state.get("active_section_focus", "").strip():
                st.caption(f"Current section focus: `{st.session_state['active_section_focus'].strip()}`.")
        elif st.session_state.get("use_track_context"):
            st.caption("No active track memory loaded for this session.")

        if st.session_state.get("last_query_response") and st.session_state["last_query_response"].has_saved:
            st.success(f"Saved to {st.session_state['last_query_response'].saved_path}")
        else:
            save_caption = f"This workflow saves by default to `{music_workflow_service.default_save_path(selected_workflow)}`."
            if active_track_context is not None:
                save_caption = (
                    f"This workflow saves by default to "
                    f"`{music_workflow_service.default_save_path(selected_workflow, track_id=active_track_context.track_id)}`."
                )
                save_caption += f" Saved answers will include Track Context metadata for `{active_track_context.track_name or active_track_context.track_id}`."
            st.caption(save_caption)

        _render_task_panel(query_service)
        chat_detail_mount = st.container()

    chat_workspace_enabled = selected_workflow.value in _CHAT_TASK_WORKFLOWS

    with main_col:
        _render_current_track_summary(active_track_context)
        critique_status = critique_support_summary(
            selected_workflow,
            active_track_context,
            (
                st.session_state.get("last_query_response").retrieved_chunks
                if isinstance(st.session_state.get("last_query_response"), QueryResponse)
                else []
            ),
        )
        if critique_status is not None:
            _render_critique_support_panel(*critique_status)
        if chat_workspace_enabled:
            st.markdown("### Session Chat")
            if active_track_context is not None:
                st.caption(
                    f"Working with active track memory for `{active_track_context.track_name or active_track_context.track_id}`. "
                    "Read the latest turn above, then reply immediately below to keep the collaboration moving."
                )
            else:
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
                    track_id=_active_yaml_track_id() or None,
                    use_track_context=st.session_state["use_track_context"],
                    track_context=active_track_context,
                    section_focus=st.session_state.get("active_section_focus", "").strip() or None,
                    chat_provider_override=_session_chat_provider_override() or None,
                    chat_model_override=_session_chat_model_override() or None,
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
                            track_context=request.track_context,
                            chat_provider_override=request.chat_provider_override,
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
                f"Active chat provider/model: `{response.debug.active_chat_provider or _effective_chat_provider(config)}` / `{response.debug.active_chat_model or _effective_chat_model(config)}`"
            )
            if response.track_context is not None:
                st.caption(
                    f"Answered with Track Context for `{response.track_context.track_name or response.track_context.track_id}` (`{response.track_context.track_id}`)."
                )
            st.write(response.answer)
            if response.track_context_update is not None and response.track_context is not None:
                base_track_context = active_track_context or response.track_context
                preview_context = query_service.track_context_update_service.preview(
                    base_track_context,
                    response.track_context_update,
                )
                st.markdown("#### Suggested Track Context Update")
                st.caption("This proposal is reviewable first, then session-applied. YAML persistence still happens only when you save Track Context from the sidebar.")
                with st.container(border=True):
                    if response.track_context_update.summary:
                        st.write(f"**Summary:** {response.track_context_update.summary}")
                    if response.track_context_update.confidence:
                        st.write(f"**Confidence:** {response.track_context_update.confidence}")
                    if response.track_context_update.source_reasoning:
                        st.write(f"**Why this was suggested:** {response.track_context_update.source_reasoning}")
                    for heading, items in proposal_groups(response.track_context_update):
                        st.markdown(f"**{heading}**")
                        for item in items:
                            st.write(f"- {item}")

                st.markdown("#### Updated Track Context Preview")
                with st.container(border=True):
                    preview_title, preview_caption, preview_rows = current_track_summary(
                        preview_context,
                        use_track_context=True,
                        track_id=preview_context.track_id,
                    )
                    st.markdown(f"**{preview_title}**")
                    st.caption(f"Read-only preview after applying the suggested update. {preview_caption}")
                    preview_left, preview_right = st.columns(2)
                    midpoint = (len(preview_rows) + 1) // 2
                    for label, value in preview_rows[:midpoint]:
                        preview_left.write(f"**{label}:** {value}")
                    for label, value in preview_rows[midpoint:]:
                        preview_right.write(f"**{label}:** {value}")

                apply_disabled = _active_yaml_track_id() == "" or response.track_context is None
                if st.button("Apply To Active Track Context", use_container_width=False, disabled=apply_disabled):
                    try:
                        updated_context = query_service.track_context_update_service.apply(
                            base_track_context,
                            response.track_context_update,
                        )
                        st.session_state["current_track_context"] = updated_context
                        if response.track_context_update.section_focus.strip():
                            st.session_state["active_section_focus"] = response.track_context_update.section_focus.strip()
                        st.session_state["track_context_editor_synced_track_id"] = ""
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
                            track_context_update=None,
                            track_context_suggestions=response.track_context_suggestions,
                        )
                        st.session_state["track_context_apply_success"] = "Track Context update applied to the active session track."
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
    track_id = _active_yaml_track_id()
    if not st.session_state.get("use_track_context") or not track_id:
        return None
    active_context = st.session_state.get("current_track_context")
    if active_context is not None and getattr(active_context, "track_id", "") == track_id:
        return active_context
    loaded_context = query_service.track_context_service.load(track_id)
    st.session_state["current_track_context"] = loaded_context
    return loaded_context


def _active_yaml_track_id() -> str:
    return st.session_state.get("active_track_context_id", "").strip()


def _render_current_track_summary(track_context) -> None:
    title, caption, rows = current_track_summary(
        track_context,
        use_track_context=st.session_state.get("use_track_context", True),
        track_id=_active_yaml_track_id(),
    )
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(caption)
        if not rows:
            return
        left_col, right_col = st.columns(2)
        midpoint = (len(rows) + 1) // 2
        for label, value in rows[:midpoint]:
            left_col.write(f"**{label}:** {value}")
        for label, value in rows[midpoint:]:
            right_col.write(f"**{label}:** {value}")


def _render_critique_support_panel(title: str, lines: list[str]) -> None:
    with st.container(border=True):
        st.markdown("### Critique Context")
        st.caption(title)
        for line in lines:
            st.write(f"- {line}")


def _sync_track_context_editor_state(track_context: TrackContext) -> None:
    synced_track_id = st.session_state.get("track_context_editor_synced_track_id", "").strip()
    if synced_track_id == track_context.track_id:
        return
    st.session_state["track_context_track_name"] = track_context.track_name or ""
    st.session_state["track_context_genre"] = track_context.genre or ""
    st.session_state["track_context_bpm"] = "" if track_context.bpm is None else str(track_context.bpm)
    st.session_state["track_context_key"] = track_context.key or ""
    st.session_state["track_context_current_stage"] = track_context.current_stage or ""
    st.session_state["track_context_current_problem"] = track_context.current_problem or ""
    st.session_state["track_context_vibe"] = ", ".join(track_context.vibe)
    st.session_state["track_context_reference_tracks"] = "\n".join(track_context.reference_tracks)
    st.session_state["track_context_known_issues"] = "\n".join(track_context.known_issues)
    st.session_state["track_context_goals"] = "\n".join(track_context.goals)
    st.session_state["track_context_editor_synced_track_id"] = track_context.track_id


def _render_ingest_tab(ingestion_service: IngestionService) -> None:
    st.caption(
        "Use ingestion to save external content into your vault as normal Markdown notes. "
        "This is separate from query-time web search, and imported notes are excluded from indexing by default."
    )

    webpage_col, youtube_col = st.columns(2)
    pdf_col, docx_col = st.columns(2)
    import_genre_service = ImportGenreService(ingestion_service.config)
    available_import_genres = import_genre_service.available_genres()
    knowledge_category_service = KnowledgeCategoryService(ingestion_service.config)
    available_knowledge_categories = knowledge_category_service.display_options()
    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("Refresh Knowledge categories", use_container_width=True):
            st.rerun()

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
            st.selectbox(
                "Genre",
                options=available_import_genres,
                key="ingest_genre",
                help="Choose the genre folder for this import. Use Generic for content that is not genre-specific.",
            )
            st.text_input(
                "Add new genre",
                key="ingest_new_genre",
                help="Optional. If filled, this overrides the dropdown and creates a new genre folder.",
            )
            st.selectbox(
                "Knowledge category",
                options=available_knowledge_categories,
                key="ingest_knowledge_category",
                help="Map this import to an existing first-level folder under Knowledge. Leave blank for generic advice.",
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
                            import_genre=_selected_import_genre("ingest"),
                            knowledge_category=_selected_knowledge_category("ingest"),
                        )
                    )
                    st.session_state["last_ingestion_response"] = response
                    st.success(f"Saved webpage note to {response.saved_path}")
                except Exception as exc:
                    st.session_state["last_ingestion_response"] = None
                    st.error(str(exc))

    with youtube_col:
        st.subheader("Import a YouTube Video")
        st.caption("Saved as a structured video knowledge note in the configured YouTube-imports folder.")
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
            st.selectbox(
                "Genre",
                options=available_import_genres,
                key="youtube_genre",
                help="Choose the genre folder for this import. Use Generic for content that is not genre-specific.",
            )
            st.text_input(
                "Add new genre",
                key="youtube_new_genre",
                help="Optional. If filled, this overrides the dropdown and creates a new genre folder.",
            )
            st.selectbox(
                "Knowledge category",
                options=available_knowledge_categories,
                key="youtube_knowledge_category",
                help="Map this import to an existing first-level folder under Knowledge. Leave blank for generic advice.",
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
                    with st.spinner(
                        "Importing video, extracting transcript, building semantic sections, and saving the note..."
                    ):
                        response = ingestion_service.ingest_youtube(
                            IngestionRequest(
                                source=url,
                                title_override=st.session_state["youtube_title"].strip() or None,
                                index_now=st.session_state["youtube_index_now"],
                                import_genre=_selected_import_genre("youtube"),
                                knowledge_category=_selected_knowledge_category("youtube"),
                            )
                        )
                    st.session_state["last_ingestion_response"] = response
                    st.success(f"Saved YouTube note to {response.saved_path}")
                except Exception as exc:
                    st.session_state["last_ingestion_response"] = None
                    st.error(str(exc))

    with pdf_col:
        st.subheader("Import a PDF")
        st.caption("Saved into the configured PDF-imports folder as a Markdown note for later retrieval.")
        with st.form("ingest_pdf_form", clear_on_submit=False, enter_to_submit=False):
            st.text_input(
                "PDF file path",
                placeholder="/path/to/document.pdf",
                key="pdf_path",
            )
            st.text_input(
                "Optional PDF note title",
                key="pdf_title",
                help="Override the saved note title and filename slug.",
            )
            st.selectbox(
                "Genre",
                options=available_import_genres,
                key="pdf_genre",
                help="Choose the genre folder for this import. Use Generic for content that is not genre-specific.",
            )
            st.text_input(
                "Add new genre",
                key="pdf_new_genre",
                help="Optional. If filled, this overrides the dropdown and creates a new genre folder.",
            )
            st.selectbox(
                "Knowledge category",
                options=available_knowledge_categories,
                key="pdf_knowledge_category",
                help="Map this import to an existing first-level folder under Knowledge. Leave blank for generic advice.",
            )
            st.checkbox(
                "Index PDF immediately",
                help="Run the existing incremental index after creating the note.",
                key="pdf_index_now",
            )
            pdf_submit = st.form_submit_button("Ingest PDF", type="primary", use_container_width=True)

        if pdf_submit:
            file_path = st.session_state["pdf_path"].strip()
            if not file_path:
                st.warning("Enter a PDF file path before starting ingestion.")
            else:
                try:
                    response = ingestion_service.ingest_pdf(
                        IngestionRequest(
                            source=file_path,
                            title_override=st.session_state["pdf_title"].strip() or None,
                            index_now=st.session_state["pdf_index_now"],
                            import_genre=_selected_import_genre("pdf"),
                            knowledge_category=_selected_knowledge_category("pdf"),
                        )
                    )
                    st.session_state["last_ingestion_response"] = response
                    st.success(f"Saved PDF note to {response.saved_path}")
                except Exception as exc:
                    st.session_state["last_ingestion_response"] = None
                    st.error(str(exc))

    with docx_col:
        st.subheader("Import a DOCX")
        st.caption("Saved into the configured Word-imports folder as a Markdown note for later retrieval.")
        with st.form("ingest_docx_form", clear_on_submit=False, enter_to_submit=False):
            st.text_input(
                "DOCX file path",
                placeholder="/path/to/document.docx",
                key="docx_path",
            )
            st.text_input(
                "Optional DOCX note title",
                key="docx_title",
                help="Override the saved note title and filename slug.",
            )
            st.selectbox(
                "Genre",
                options=available_import_genres,
                key="docx_genre",
                help="Choose the genre folder for this import. Use Generic for content that is not genre-specific.",
            )
            st.text_input(
                "Add new genre",
                key="docx_new_genre",
                help="Optional. If filled, this overrides the dropdown and creates a new genre folder.",
            )
            st.selectbox(
                "Knowledge category",
                options=available_knowledge_categories,
                key="docx_knowledge_category",
                help="Map this import to an existing first-level folder under Knowledge. Leave blank for generic advice.",
            )
            st.checkbox(
                "Index DOCX immediately",
                help="Run the existing incremental index after creating the note.",
                key="docx_index_now",
            )
            docx_submit = st.form_submit_button("Ingest DOCX", type="primary", use_container_width=True)

        if docx_submit:
            file_path = st.session_state["docx_path"].strip()
            if not file_path:
                st.warning("Enter a DOCX file path before starting ingestion.")
            else:
                try:
                    response = ingestion_service.ingest_docx(
                        IngestionRequest(
                            source=file_path,
                            title_override=st.session_state["docx_title"].strip() or None,
                            index_now=st.session_state["docx_index_now"],
                            import_genre=_selected_import_genre("docx"),
                            knowledge_category=_selected_knowledge_category("docx"),
                        )
                    )
                    st.session_state["last_ingestion_response"] = response
                    st.success(f"Saved DOCX note to {response.saved_path}")
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
    if response.import_genre:
        st.write(f"Genre: `{response.import_genre}`")
    if response.knowledge_category:
        st.write(f"Knowledge category: `{response.knowledge_category}`")
    if response.section_count:
        st.write(f"Sections: `{response.section_count}`")
    if response.transcript_chunk_count:
        st.write(f"Transcript chunks: `{response.transcript_chunk_count}`")
    st.write(f"Indexed now: `{'yes' if response.index_triggered else 'no'}`")
    for warning in response.warnings:
        st.warning(warning)


def _selected_import_genre(prefix: str) -> str:
    new_genre = st.session_state.get(f"{prefix}_new_genre", "").strip()
    if new_genre:
        return new_genre
    return st.session_state.get(f"{prefix}_genre", GENERIC_IMPORT_GENRE)


def _selected_knowledge_category(prefix: str) -> str | None:
    selected = st.session_state.get(
        f"{prefix}_knowledge_category",
        GENERIC_KNOWLEDGE_CATEGORY_LABEL,
    )
    if selected == GENERIC_KNOWLEDGE_CATEGORY_LABEL:
        return None
    return selected


def _render_debug_section(response: QueryResponse, original_question: str) -> None:
    with st.expander("Debug Details", expanded=False):
        st.markdown("**Query Summary**")
        for label, value in debug_query_summary(original_question, response.debug.rewritten_query):
            st.write(f"{label}: `{value}`")
        if response.debug.imported_genres_eligible:
            st.write(
                "Imported genres eligible for retrieval: "
                + ", ".join(f"`{genre}`" for genre in response.debug.imported_genres_eligible)
            )

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
                "response_mode_selected": response.debug.response_mode_selected,
                "followup_triggered": response.debug.followup_triggered,
                "missing_dimension": response.debug.missing_dimension,
                "active_section": response.debug.active_section,
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
        metric_cols[3].metric("Embeddings Backend", "Reachable" if status.ollama_reachable else "Unavailable")

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
    active_chat_provider = _effective_chat_provider(config)
    available_chat_models, _ = list_available_chat_models(
        config,
        provider_override=_session_chat_provider_override() or None,
    )
    active_chat_model = (
        _resolve_preferred_chat_model_name(_effective_chat_model(config), available_chat_models)
        if active_chat_provider == "ollama" and available_chat_models
        else _effective_chat_model(config)
    )
    st.subheader("Active Models")
    model_cols = st.columns(2)
    model_cols[0].write(f"Chat provider/model: `{active_chat_provider}` / `{active_chat_model}`")
    model_cols[1].write(f"Embedding provider/model: `{config.embedding_provider}` / `{configured_embedding_model(config)}`")
    if _session_dev_mode_preset():
        st.caption(f"Active dev preset: `{_session_dev_mode_preset()}`.")
    else:
        st.caption("Active dev preset: `manual`.")
    if _session_chat_provider_override().strip():
        st.caption(
            f"Session provider override active. Configured default provider: `{config.chat_provider}`; current provider override: `{_session_chat_provider_override()}`."
        )
    else:
        st.caption(f"No session provider override active. Using configured default chat provider: `{config.chat_provider}`.")
    if _session_chat_model_override().strip():
        st.caption(
            f"Session model override active. Configured default model for `{active_chat_provider}`: `{configured_chat_model(config, provider_override=_session_chat_provider_override() or None) or '(not configured)'}`; current override: `{_session_chat_model_override()}`."
        )
    else:
        st.caption(
            f"No session model override active. Using configured default chat model for `{active_chat_provider}`: `{configured_chat_model(config, provider_override=_session_chat_provider_override() or None) or '(not configured)'}`."
        )
    st.caption("Chat provider/model overrides affect chat generation only. Embeddings remain Ollama-based for now.")

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
            "By default, indexing excludes saved outputs, saved research outputs, webpage imports, and YouTube imports. "
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


def _apply_legacy_track_selection(
    config: AppConfig,
    selected_track: str,
    tracks: list[dict[str, str]],
) -> None:
    current_path = st.session_state.get("workflow_track_context_path", "").strip()
    resolved_path = selected_track_path(selected_track, tracks)
    if resolved_path is None:
        if current_path in {track["path"] for track in tracks}:
            st.session_state["workflow_track_context_path"] = ""
        st.session_state["workflow_track_selector_applied_path"] = ""
        return
    if current_path != resolved_path:
        st.session_state["workflow_track_context_path"] = resolved_path

    if st.session_state.get("workflow_track_selector_applied_path", "") == resolved_path:
        return

    autofill_values = TrackSelectorService().load_workflow_context(
        config.obsidian_vault_path,
        resolved_path,
    )
    for key, value in autofill_values.items():
        st.session_state[key] = value
    st.session_state["workflow_track_selector_applied_path"] = resolved_path


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


def _render_task_panel(query_service: QueryService) -> None:
    active_track_id = _active_yaml_track_id() or ""
    persisted_mode = bool(st.session_state.get("use_track_context") and active_track_id)
    panel_title = "Track Tasks" if persisted_mode else "Session Tasks"
    with st.expander(panel_title, expanded=False):
        if persisted_mode:
            st.caption(
                "Tasks are persisted per active YAML track and used as internal execution context for critique, arrangement, and sound-design workflows."
            )
        else:
            st.caption(
                "Tasks are session-only when no active YAML track is loaded and are used as internal execution context for critique, arrangement, and sound-design workflows."
            )
        with st.form("add_task_form", clear_on_submit=False, enter_to_submit=False):
            st.text_input("Task", key="new_task_text", placeholder="Add a focused production task")
            st.text_input("Notes (optional)", key="new_task_notes", placeholder="Optional detail")
            add_task = st.form_submit_button("Add Task", use_container_width=True)

        if add_task and st.session_state["new_task_text"].strip():
            if persisted_mode:
                query_service.track_task_service.add_task(
                    active_track_id,
                    text=st.session_state["new_task_text"].strip(),
                    created_from="user",
                    linked_section=st.session_state.get("active_section_focus", "").strip(),
                    notes=st.session_state["new_task_notes"].strip(),
                )
                st.session_state["session_tasks"] = query_service.track_task_service.load_session_tasks(active_track_id)
                st.session_state["session_tasks_track_id"] = active_track_id
            else:
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
            st.caption("No open tasks.")
        for task in open_tasks:
            _render_task_actions(task, query_service=query_service)

        st.markdown("**Completed Tasks**")
        if not completed_tasks:
            st.caption("No completed tasks yet.")
        for task in completed_tasks:
            _render_task_actions(task, query_service=query_service)


def _render_task_actions(task: SessionTask, *, query_service: QueryService) -> None:
    active_track_id = _active_yaml_track_id() or ""
    persisted_mode = bool(st.session_state.get("use_track_context") and active_track_id)
    cols = st.columns([4, 1, 1])
    meta_parts: list[str] = []
    if task.priority.strip() and task.priority.strip().lower() != "medium":
        meta_parts.append(task.priority.strip())
    if task.linked_section.strip():
        meta_parts.append(task.linked_section.strip())
    meta_suffix = f" [{' | '.join(meta_parts)}]" if meta_parts else ""
    note_suffix = f" ({task.notes})" if task.notes else ""
    cols[0].write(f"{'[ ]' if task.status == 'open' else '[x]'} {task.text}{meta_suffix}{note_suffix}")
    toggle_label = "Done" if task.status == "open" else "Reopen"
    if cols[1].button(toggle_label, key=f"task-toggle-{task.id}", use_container_width=True):
        if persisted_mode:
            query_service.track_task_service.complete_task(
                active_track_id,
                task.id,
                completed=task.status == "open",
            )
            st.session_state["session_tasks"] = query_service.track_task_service.load_session_tasks(active_track_id)
            st.session_state["session_tasks_track_id"] = active_track_id
        else:
            updated_tasks: list[SessionTask] = []
            for existing_task in st.session_state["session_tasks"]:
                if existing_task.id == task.id:
                    updated_tasks.append(
                        replace(
                            existing_task,
                            status="completed" if existing_task.status == "open" else "open",
                            completed_at=current_timestamp() if existing_task.status == "open" else None,
                        )
                    )
                else:
                    updated_tasks.append(existing_task)
            st.session_state["session_tasks"] = updated_tasks
        st.rerun()
    if cols[2].button("Delete", key=f"task-delete-{task.id}", use_container_width=True):
        if persisted_mode:
            query_service.track_task_service.delete_task(active_track_id, task.id)
            st.session_state["session_tasks"] = query_service.track_task_service.load_session_tasks(active_track_id)
            st.session_state["session_tasks_track_id"] = active_track_id
        else:
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
        "dev_mode_preset_select": DEV_MODE_PRESET_MANUAL,
        "dev_mode_preset": "",
        "last_synced_dev_mode_preset": "",
        "active_chat_provider_select": f"Use configured default ({config.chat_provider})",
        "last_synced_chat_provider_override": "",
        "active_chat_model_select": _default_active_chat_model(config),
        "active_chat_model_input": "",
        "chat_provider_override": "",
        "chat_model_override": "",
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
        "workflow_track_selector_applied_path": "",
        "track_context_track_id": "",
        "active_track_context_id": "",
        "active_track_context_loaded_existing": False,
        "current_track_context": None,
        "active_section_focus": "",
        "track_context_editor_synced_track_id": "",
        "use_track_context": True,
        "track_context_track_name": "",
        "track_context_genre": "",
        "track_context_bpm": "",
        "track_context_key": "",
        "track_context_current_stage": "",
        "track_context_current_problem": "",
        "track_context_vibe": "",
        "track_context_reference_tracks": "",
        "track_context_known_issues": "",
        "track_context_goals": "",
        "chat_messages": [],
        "session_tasks": [],
        "session_tasks_track_id": "",
        "new_task_text": "",
        "new_task_notes": "",
        "max_subquestions": 3,
        "debug_mode": False,
        "last_query_response": None,
        "last_question": "",
        "ingest_url": "",
        "ingest_title": "",
        "ingest_genre": GENERIC_IMPORT_GENRE,
        "ingest_new_genre": "",
        "ingest_knowledge_category": GENERIC_KNOWLEDGE_CATEGORY_LABEL,
        "ingest_index_now": config.auto_index_after_ingestion,
        "youtube_url": "",
        "youtube_title": "",
        "youtube_genre": GENERIC_IMPORT_GENRE,
        "youtube_new_genre": "",
        "youtube_knowledge_category": GENERIC_KNOWLEDGE_CATEGORY_LABEL,
        "youtube_index_now": config.auto_index_after_ingestion,
        "pdf_path": "",
        "pdf_title": "",
        "pdf_genre": GENERIC_IMPORT_GENRE,
        "pdf_new_genre": "",
        "pdf_knowledge_category": GENERIC_KNOWLEDGE_CATEGORY_LABEL,
        "pdf_index_now": config.auto_index_after_ingestion,
        "docx_path": "",
        "docx_title": "",
        "docx_genre": GENERIC_IMPORT_GENRE,
        "docx_new_genre": "",
        "docx_knowledge_category": GENERIC_KNOWLEDGE_CATEGORY_LABEL,
        "docx_index_now": config.auto_index_after_ingestion,
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


def _sync_active_chat_model_with_available_models(config: AppConfig) -> None:
    active_provider_override = _session_chat_provider_override() or None
    available_chat_models, _ = list_available_chat_models(
        config,
        provider_override=active_provider_override,
    )
    if not available_chat_models:
        st.session_state["active_chat_model_input"] = st.session_state.get("chat_model_override", "").strip()
        return

    override_model = st.session_state.get("chat_model_override", "").strip()
    if override_model:
        st.session_state["active_chat_model_select"] = _resolve_preferred_chat_model_name(
            override_model,
            available_chat_models,
        )
    else:
        st.session_state["active_chat_model_select"] = (
            f"Use configured default ({configured_chat_model(config, provider_override=active_provider_override)})"
        )
    st.session_state["active_chat_model_input"] = override_model


def _sync_active_chat_provider_with_session(config: AppConfig) -> None:
    selection, synced_override = synced_chat_provider_selection(
        current_selection=st.session_state.get("active_chat_provider_select", ""),
        committed_override=_session_chat_provider_override(),
        configured_provider=config.chat_provider,
        last_synced_override=st.session_state.get("last_synced_chat_provider_override", ""),
    )
    st.session_state["active_chat_provider_select"] = selection
    st.session_state["last_synced_chat_provider_override"] = synced_override


def _sync_dev_mode_preset_with_session() -> None:
    selection, synced_preset = synced_dev_mode_preset_selection(
        current_selection=st.session_state.get("dev_mode_preset_select", ""),
        committed_preset=_session_dev_mode_preset(),
        last_synced_preset=st.session_state.get("last_synced_dev_mode_preset", ""),
    )
    st.session_state["dev_mode_preset_select"] = selection
    st.session_state["last_synced_dev_mode_preset"] = synced_preset


_CHAT_TASK_WORKFLOWS = {
    CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE.value,
    CollaborationWorkflow.ARRANGEMENT_PLANNER.value,
    CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM.value,
}

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

def _default_active_chat_model(config: AppConfig) -> str:
    if _effective_chat_provider(config) == "openai":
        return configured_chat_model(
            config,
            provider_override=_session_chat_provider_override() or None,
        ) or "openai"
    return _DEFAULT_ACTIVE_CHAT_MODEL


def _effective_chat_model(config: AppConfig) -> str:
    return (
        _session_chat_model_override()
        or configured_chat_model(config, provider_override=_session_chat_provider_override() or None)
        or _DEFAULT_ACTIVE_CHAT_MODEL
    )


def _effective_chat_provider(config: AppConfig) -> str:
    return effective_chat_provider(config, provider_override=_session_chat_provider_override() or None)


def _session_dev_mode_preset() -> str:
    return st.session_state.get("dev_mode_preset", "").strip()


def _session_chat_provider_override() -> str:
    return st.session_state.get("chat_provider_override", "").strip()


def _session_chat_model_override() -> str:
    return st.session_state.get("chat_model_override", "").strip()


def _ordered_chat_provider_options(default_provider: str) -> list[str]:
    providers = [provider for provider in _CHAT_PROVIDER_OPTIONS if provider != default_provider]
    return [default_provider, *providers]


_DEFAULT_ACTIVE_CHAT_MODEL = "deepseek"
_CHAT_PROVIDER_OPTIONS = ("openai", "ollama")


if __name__ == "__main__":
    main()
