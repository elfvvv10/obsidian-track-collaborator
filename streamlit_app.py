"""Local Streamlit UI for the Obsidian RAG assistant."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from config import AppConfig, load_config
from services.ingestion_service import IngestionService
from services.index_service import IndexService
from services.models import (
    AnswerMode,
    IngestionRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    ResearchResponse,
    RetrievalMode,
    WorkflowMode,
)
from services.query_service import QueryService
from services.research_service import ResearchService
from utils import RetrievalFilters, RetrievalOptions


st.set_page_config(page_title="Obsidian RAG Assistant", layout="wide")


def main() -> None:
    """Render the Streamlit app."""
    st.title("Obsidian RAG Assistant")
    st.caption("Local-only research assistant for your Obsidian vault.")

    try:
        base_config = load_config()
    except Exception as exc:
        st.error(str(exc))
        return

    _init_session_state(base_config)
    ui_config = _config_from_session(base_config)
    services = _get_services(ui_config)
    status, status_error = _safe_get_status(services["index_service"])

    _render_sidebar(base_config, status, status_error)

    ask_tab, ingest_tab, index_tab, settings_tab = st.tabs(
        ["Ask", "Ingest", "Index", "Settings / Debug"]
    )

    with ask_tab:
        _render_ask_tab(ui_config, services["query_service"], services["research_service"], status)

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
    }


def _safe_get_status(index_service: IndexService) -> tuple[IndexResponse | None, str | None]:
    try:
        return index_service.get_status(), None
    except Exception as exc:
        return None, str(exc)


def _render_sidebar(config: AppConfig, status: IndexResponse | None, status_error: str | None) -> None:
    with st.sidebar:
        st.header("Ask Controls")
        st.session_state["folder_filter"] = st.text_input(
            "Folder filter",
            value=st.session_state["folder_filter"],
            help="Only search notes from this vault-relative folder.",
        )
        st.session_state["path_filter"] = st.text_input(
            "Path contains",
            value=st.session_state["path_filter"],
            help="Only search notes whose path contains this text.",
        )
        st.session_state["tag_filter"] = st.text_input(
            "Tag filter",
            value=st.session_state["tag_filter"],
            help="Only search notes with this tag.",
        )
        st.session_state["top_k"] = int(
            st.number_input(
                "Top-k",
                min_value=1,
                max_value=20,
                value=int(st.session_state["top_k"]),
                help="How many final chunks to use when answering.",
            )
        )
        st.session_state["enable_reranking"] = st.checkbox(
            "Enable reranking",
            value=st.session_state["enable_reranking"],
        )
        st.session_state["include_linked"] = st.checkbox(
            "Include linked-note context",
            value=st.session_state["include_linked"],
        )
        st.session_state["auto_save"] = st.checkbox(
            "Auto-save answers",
            value=st.session_state["auto_save"],
        )
        st.session_state["retrieval_mode"] = st.selectbox(
            "Retrieval mode",
            options=[mode.value for mode in RetrievalMode],
            index=[mode.value for mode in RetrievalMode].index(st.session_state["retrieval_mode"]),
            help="Choose local-only, automatic web fallback, or always-on hybrid web use.",
        )
        st.session_state["answer_mode"] = st.selectbox(
            "Answer mode",
            options=[mode.value for mode in AnswerMode],
            index=[mode.value for mode in AnswerMode].index(st.session_state["answer_mode"]),
            help=(
                "Strict: retrieved evidence only. "
                "Balanced: evidence first with limited reasoning. "
                "Exploratory: evidence plus broader synthesis and labeled inference."
            ),
        )

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
    status: IndexResponse | None,
) -> None:
    if status is not None:
        if status.total_chunks_stored == 0:
            st.info("Build the index in the Index tab before asking questions.")
        elif not status.index_compatible:
            st.warning("The local index is out of date. Rebuild it from the Index tab.")

    question_col, save_col = st.columns([3, 2])
    with question_col:
        question = st.text_area(
            "Question",
            value=st.session_state.get("question", ""),
            placeholder="What do my notes say about AI agents?",
            height=120,
        )
        st.session_state["question"] = question
        with st.container(border=True):
            st.markdown("#### Workflow")
            st.session_state["workflow_mode"] = st.radio(
                "Question workflow",
                options=[WorkflowMode.DIRECT.value, WorkflowMode.RESEARCH.value],
                index=[WorkflowMode.DIRECT.value, WorkflowMode.RESEARCH.value].index(
                    st.session_state["workflow_mode"]
                ),
                horizontal=True,
                format_func=lambda value: "Direct Ask" if value == WorkflowMode.DIRECT.value else "Research Mode",
            )
            if st.session_state["workflow_mode"] == WorkflowMode.RESEARCH.value:
                st.session_state["max_subquestions"] = int(
                    st.number_input(
                        "Max research subquestions",
                        min_value=1,
                        max_value=5,
                        value=int(st.session_state["max_subquestions"]),
                        help="Research mode decomposes your request into a small visible set of subquestions.",
                    )
                )
            st.markdown("#### Retrieval Scope")
            st.session_state["use_saved_answers"] = st.toggle(
                "Use saved answers for this question",
                value=st.session_state["use_saved_answers"],
                help="Include previously saved answer notes as secondary sources for this question only.",
            )
            if st.session_state["use_saved_answers"]:
                if config.index_saved_answers:
                    st.caption("Saved answers are enabled for this question and will be treated as secondary sources.")
                else:
                    st.warning(
                        "Saved answers are not currently indexed. Enable `INDEX_SAVED_ANSWERS=true` and rebuild the index to use them here."
                    )

    with save_col:
        st.markdown("### Save Options")
        st.session_state["save_title"] = st.text_input(
            "Optional note title",
            value=st.session_state["save_title"],
            help="Override the saved note title and filename slug.",
        )
        if st.session_state.get("last_query_response") and st.session_state["last_query_response"].has_saved:
            st.success(f"Saved to {st.session_state['last_query_response'].saved_path}")
        else:
            st.caption("You can save the current answer after asking a question.")

    ask_clicked = st.button("Ask", type="primary")
    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question before asking.")
        else:
            try:
                request = QueryRequest(
                    question=question.strip(),
                    filters=_current_filters(),
                    options=_current_options(),
                    auto_save=st.session_state["auto_save"],
                    save_title=st.session_state["save_title"].strip() or None,
                    retrieval_mode=st.session_state["retrieval_mode"],
                    answer_mode=st.session_state["answer_mode"],
                )
                if st.session_state["workflow_mode"] == WorkflowMode.RESEARCH.value:
                    response = research_service.research(
                        ResearchRequest(
                            goal=question.strip(),
                            filters=request.filters,
                            options=request.options,
                            auto_save=request.auto_save,
                            save_title=request.save_title,
                            retrieval_mode=request.retrieval_mode,
                            answer_mode=request.answer_mode,
                            max_subquestions=st.session_state["max_subquestions"],
                        )
                    )
                else:
                    response = query_service.ask(request)
                st.session_state["last_query_response"] = response
                st.session_state["last_question"] = question.strip()
            except Exception as exc:
                st.session_state["last_query_response"] = None
                st.error(str(exc))

    response = st.session_state.get("last_query_response")
    if response is None:
        return

    if isinstance(response, ResearchResponse):
        _render_research_response(question, response, research_service)
        return

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
    status_cols[2].metric("Local notes used", "Yes" if response.local_sources else "No")
    status_cols[3].metric("Web used", "Yes" if response.web_used else "No")
    status_cols[4].metric("Inference used", "Yes" if response.inference_used else "No")

    summary_col, sources_col = st.columns([3, 2])
    with summary_col:
        st.subheader("Answer")
        st.write(response.answer)
    with sources_col:
        st.subheader("Sources")
        if response.local_sources:
            for source in response.local_sources:
                st.write(f"- {source}")
        else:
            st.write("- No local note sources")
        if response.saved_sources:
            st.markdown("**Saved Answer Sources**")
            for source in response.saved_sources:
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
        _render_debug_section(response)


def _render_research_response(
    question: str,
    response: ResearchResponse,
    research_service: ResearchService,
) -> None:
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
    status_cols[2].metric("Local notes used", "Yes" if response.local_sources else "No")
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
        if response.local_sources:
            for source in response.local_sources:
                st.write(f"- {source}")
        else:
            st.write("- No local note sources")
        if response.saved_sources:
            st.markdown("**Saved Answer Sources**")
            for source in response.saved_sources:
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


def _render_ingest_tab(ingestion_service: IngestionService) -> None:
    st.caption(
        "Use ingestion to save external content into your vault as normal Markdown notes. "
        "This is separate from query-time web search."
    )

    webpage_col, youtube_col = st.columns(2)

    with webpage_col:
        st.subheader("Import a Webpage")
        st.session_state["ingest_url"] = st.text_input(
            "Webpage URL",
            value=st.session_state["ingest_url"],
            placeholder="https://example.com/article",
            key="ingest_url_input",
        )
        st.session_state["ingest_title"] = st.text_input(
            "Optional note title",
            value=st.session_state["ingest_title"],
            help="Override the saved note title and filename slug.",
            key="ingest_title_input",
        )
        st.session_state["ingest_index_now"] = st.checkbox(
            "Index immediately after save",
            value=st.session_state["ingest_index_now"],
            help="Run the existing incremental index after creating the note.",
            key="ingest_index_now_checkbox",
        )

        if st.button("Ingest Webpage", type="primary"):
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
        st.session_state["youtube_url"] = st.text_input(
            "YouTube URL",
            value=st.session_state["youtube_url"],
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
        )
        st.session_state["youtube_title"] = st.text_input(
            "Optional video note title",
            value=st.session_state["youtube_title"],
            help="Override the saved note title and filename slug.",
            key="youtube_title_input",
        )
        st.session_state["youtube_index_now"] = st.checkbox(
            "Index YouTube note immediately",
            value=st.session_state["youtube_index_now"],
            help="Run the existing incremental index after creating the note.",
            key="youtube_index_now_checkbox",
        )

        if st.button("Ingest YouTube", type="primary"):
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


def _render_debug_section(response: QueryResponse) -> None:
    with st.expander("Debug Details", expanded=False):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Initial candidates", len(response.debug.initial_candidates))
        metric_cols[1].metric("Primary chunks", len(response.debug.primary_chunks))
        metric_cols[2].metric("Final chunks", len(response.retrieved_chunks))
        metric_cols[3].metric("Reranking changed order", "Yes" if response.debug.reranking_changed else "No")

        st.markdown("**Retrieval Settings**")
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
                "retrieval_mode_requested": response.debug.retrieval_mode_requested,
                "retrieval_mode_used": response.debug.retrieval_mode_used,
                "answer_mode_requested": response.debug.answer_mode_requested,
                "answer_mode_used": response.debug.answer_mode_used,
                "web_used": response.debug.web_used,
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
    st.subheader("Active Models")
    model_cols = st.columns(2)
    model_cols[0].write(f"Chat model: `{config.ollama_chat_model}`")
    model_cols[1].write(f"Embedding model: `{config.ollama_embedding_model}`")

    st.subheader("Paths and Status")
    if status is None:
        st.warning("Status is currently unavailable.")
        if status_error:
            st.caption(status_error)
    else:
        st.write(f"Vault path: `{status.vault_path}`")
        st.write(f"Output path: `{status.output_path}`")
        st.write(f"Webpage ingestion folder: `{config.webpage_ingestion_folder}`")
        st.write(f"YouTube ingestion folder: `{config.youtube_ingestion_folder}`")
        st.write(f"Index schema version: `{status.index_version or 'not set'}`")
        st.write(f"Stored chunks: `{status.total_chunks_stored}`")
        st.write(f"Index compatible: `{'yes' if status.index_compatible else 'no'}`")
        st.write(f"App ready: `{'yes' if status.ready else 'no'}`")

    st.subheader("Advanced / Debug")
    st.session_state["debug_mode"] = st.checkbox(
        "Show debug details in Ask results",
        value=st.session_state["debug_mode"],
    )
    st.caption(
        "Query filters and retrieval controls live in the sidebar so they stay close to the Ask workflow."
    )
    st.write(f"Current retrieval mode: `{st.session_state['retrieval_mode']}`")
    st.write(f"Current answer mode: `{st.session_state['answer_mode']}`")
    st.write(f"Current workflow mode: `{st.session_state['workflow_mode']}`")


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
        include_saved_answers=st.session_state["use_saved_answers"],
    )


def _init_session_state(config: AppConfig) -> None:
    defaults = {
        "question": "",
        "folder_filter": "",
        "path_filter": "",
        "tag_filter": "",
        "save_title": "",
        "top_k": config.top_k_results,
        "enable_reranking": config.enable_reranking,
        "include_linked": config.enable_linked_note_expansion,
        "use_saved_answers": False,
        "auto_save": config.auto_save_answer,
        "retrieval_mode": RetrievalMode.LOCAL_ONLY.value,
        "answer_mode": AnswerMode.BALANCED.value,
        "workflow_mode": WorkflowMode.DIRECT.value,
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


if __name__ == "__main__":
    main()
