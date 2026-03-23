"""Local Streamlit UI for the Obsidian RAG assistant."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from config import AppConfig, load_config
from services.index_service import IndexService
from services.models import QueryRequest
from services.query_service import QueryService
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

    ask_tab, index_tab, settings_tab = st.tabs(["Ask", "Index", "Settings / Debug"])

    with ask_tab:
        _render_ask_tab(ui_config)

    with index_tab:
        _render_index_tab(ui_config)

    with settings_tab:
        _render_settings_tab(base_config)


def _render_ask_tab(config: AppConfig) -> None:
    question = st.text_area(
        "Question",
        value=st.session_state.get("question", ""),
        placeholder="What do my notes say about AI agents?",
        height=120,
    )
    st.session_state["question"] = question

    ask_clicked = st.button("Ask", type="primary")
    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question before asking.")
        else:
            try:
                request = QueryRequest(
                    question=question.strip(),
                    filters=RetrievalFilters(
                        tag=st.session_state["tag_filter"].strip().lstrip("#").lower() or None,
                        folder=st.session_state["folder_filter"].strip().strip("/") or None,
                        path_contains=st.session_state["path_filter"].strip().lower() or None,
                    ),
                    options=RetrievalOptions(
                        top_k=st.session_state["top_k"],
                        rerank=st.session_state["enable_reranking"],
                        include_linked_notes=st.session_state["include_linked"],
                    ),
                    auto_save=st.session_state["auto_save"],
                )
                response = QueryService(config).ask(request)
                st.session_state["last_query_response"] = response
                st.session_state["last_question"] = question.strip()
            except Exception as exc:
                st.session_state["last_query_response"] = None
                st.error(str(exc))

    response = st.session_state.get("last_query_response")
    if response is None:
        return

    if response.warnings:
        for warning in response.warnings:
            st.warning(warning)

    st.subheader("Answer")
    st.write(response.answer)

    st.subheader("Sources")
    if response.sources:
        for source in response.sources:
            st.write(f"- {source}")
    else:
        st.write("No sources retrieved.")

    if response.linked_context_chunks:
        st.subheader("Linked Note Context")
        linked_titles = sorted(
            {
                f"{chunk.metadata.get('note_title', 'Untitled')} ({chunk.metadata.get('source_path', 'unknown')})"
                for chunk in response.linked_context_chunks
            }
        )
        for linked_title in linked_titles:
            st.write(f"- {linked_title}")

    if response.saved_path is not None:
        st.success(f"Saved answer to {response.saved_path}")
    elif st.button("Save To Vault"):
        try:
            saved_response = QueryService(config).save(
                st.session_state.get("last_question", question.strip()),
                response.answer_result,
            )
            st.session_state["last_query_response"] = saved_response
            st.success(f"Saved answer to {saved_response.saved_path}")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state["debug_mode"]:
        with st.expander("Debug Details", expanded=False):
            st.write("Retrieved Chunks")
            for index, chunk in enumerate(response.retrieved_chunks, start=1):
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


def _render_index_tab(config: AppConfig) -> None:
    index_service = IndexService(config)
    try:
        status = index_service.get_status()
        st.write(f"Stored chunks: {status.total_chunks_stored}")
        st.write(f"Index compatible: {'Yes' if status.index_compatible else 'No'}")
    except Exception as exc:
        st.warning(str(exc))

    col1, col2 = st.columns(2)
    if col1.button("Build Index", type="primary"):
        try:
            response = index_service.index(reset_store=False)
            st.success(
                f"Index complete. Notes: {response.notes_loaded}, "
                f"chunks created: {response.chunks_created}, chunks indexed: {response.chunks_indexed}, "
                f"stored chunks: {response.total_chunks_stored}"
            )
            for warning in response.warnings:
                st.warning(warning)
        except Exception as exc:
            st.error(str(exc))

    if col2.button("Rebuild Index"):
        try:
            response = index_service.index(reset_store=True)
            st.success(
                f"Rebuild complete. Notes: {response.notes_loaded}, "
                f"chunks created: {response.chunks_created}, chunks indexed: {response.chunks_indexed}, "
                f"stored chunks: {response.total_chunks_stored}"
            )
            for warning in response.warnings:
                st.warning(warning)
        except Exception as exc:
            st.error(str(exc))


def _render_settings_tab(config: AppConfig) -> None:
    st.subheader("Active Configuration")
    st.write(f"Chat model: `{config.ollama_chat_model}`")
    st.write(f"Embedding model: `{config.ollama_embedding_model}`")
    st.write(f"Top-k: `{st.session_state['top_k']}`")
    st.write(f"Reranking: `{'on' if st.session_state['enable_reranking'] else 'off'}`")
    st.write(f"Linked-note expansion: `{'on' if st.session_state['include_linked'] else 'off'}`")
    st.write(f"Auto-save: `{'on' if st.session_state['auto_save'] else 'off'}`")
    st.write(f"Debug mode: `{'on' if st.session_state['debug_mode'] else 'off'}`")

    st.markdown("### Interactive Controls")
    st.session_state["folder_filter"] = st.text_input("Folder filter", value=st.session_state["folder_filter"])
    st.session_state["path_filter"] = st.text_input("Path contains", value=st.session_state["path_filter"])
    st.session_state["tag_filter"] = st.text_input("Tag filter", value=st.session_state["tag_filter"])
    st.session_state["top_k"] = st.number_input("Top-k", min_value=1, max_value=20, value=st.session_state["top_k"])
    st.session_state["enable_reranking"] = st.checkbox("Enable reranking", value=st.session_state["enable_reranking"])
    st.session_state["include_linked"] = st.checkbox("Include linked-note context", value=st.session_state["include_linked"])
    st.session_state["auto_save"] = st.checkbox("Auto-save answers", value=st.session_state["auto_save"])
    st.session_state["debug_mode"] = st.checkbox("Show debug details", value=st.session_state["debug_mode"])


def _init_session_state(config: AppConfig) -> None:
    defaults = {
        "question": "",
        "folder_filter": "",
        "path_filter": "",
        "tag_filter": "",
        "top_k": config.top_k_results,
        "enable_reranking": config.enable_reranking,
        "include_linked": config.enable_linked_note_expansion,
        "auto_save": config.auto_save_answer,
        "debug_mode": False,
        "last_query_response": None,
        "last_question": "",
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
