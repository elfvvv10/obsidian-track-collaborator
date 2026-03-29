"""CLI entrypoint for Obsidian Track Collaborator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys

from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from retriever import Retriever
from config import AppConfig, load_config
from saver import prompt_to_save, save_answer
from services.common import build_note_alias_map, ensure_index_compatible, resolve_note_links
from services.ingestion_service import IngestionService
from services.index_service import IndexService
from services.models import (
    AnswerMode,
    IngestionRequest,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    RetrievalMode,
    RetrievalScope,
)
from services.query_service import QueryService
from services.research_service import ResearchService
from services.track_context_update_review import proposal_groups
from services.web_search_service import WebSearchService
from utils import Note
from utils import RetrievalFilters, RetrievalOptions, get_logger


logger = get_logger()


def main() -> int:
    """Run the CLI application."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        config = load_config()
        if args.command == "index":
            run_index(_config_with_index_overrides(config, args), reset_store=False)
        elif args.command == "rebuild":
            run_index(_config_with_index_overrides(config, args), reset_store=True)
        elif args.command == "ask":
            run_ask(
                config,
                args.question,
                folder=args.folder,
                path_contains=args.path_contains,
                tag=args.tag,
                boost_tags=args.boost_tag,
                include_linked=args.include_linked,
                top_k=args.top_k,
                candidate_count=args.candidate_count,
                rerank=args.rerank,
                auto_save=args.auto_save,
                retrieval_scope=args.retrieval_scope,
                retrieval_mode=args.retrieval_mode,
                answer_mode=args.answer_mode,
                track_id=args.track_id,
                use_track_context=args.use_track_context,
                section_focus=args.section_focus,
            )
        elif args.command == "research":
            run_research(
                config,
                args.question,
                folder=args.folder,
                path_contains=args.path_contains,
                tag=args.tag,
                boost_tags=args.boost_tag,
                include_linked=args.include_linked,
                top_k=args.top_k,
                candidate_count=args.candidate_count,
                rerank=args.rerank,
                auto_save=args.auto_save,
                retrieval_scope=args.retrieval_scope,
                retrieval_mode=args.retrieval_mode,
                answer_mode=args.answer_mode,
                max_subquestions=args.max_subquestions,
            )
        elif args.command == "ingest-webpage":
            run_ingest_webpage(
                config,
                args.url,
                title=args.title,
                index_now=args.index_now,
            )
        elif args.command == "ingest-youtube":
            run_ingest_youtube(
                config,
                args.url,
                title=args.title,
                index_now=args.index_now,
                genre=args.genre,
            )
        elif args.command == "ingest-pdf":
            run_ingest_pdf(
                config,
                args.file_path,
                title=args.title,
                index_now=args.index_now,
                genre=args.genre,
            )
        elif args.command == "ingest-docx":
            run_ingest_docx(
                config,
                args.file_path,
                title=args.title,
                index_now=args.index_now,
                genre=args.genre,
            )
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        logger.error(str(exc))
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Obsidian Track Collaborator")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("index", help="Build the vector index from the configured vault")
    rebuild_parser = subparsers.add_parser("rebuild", help="Clear and rebuild the vector index")
    ingest_parser = subparsers.add_parser("ingest-webpage", help="Import a webpage into the vault")
    ingest_youtube_parser = subparsers.add_parser("ingest-youtube", help="Import a YouTube video knowledge note into the vault")
    ingest_pdf_parser = subparsers.add_parser("ingest-pdf", help="Import a PDF document into the vault")
    ingest_docx_parser = subparsers.add_parser("ingest-docx", help="Import a DOCX document into the vault")

    index_parser = subparsers.choices["index"]
    _add_index_overrides(index_parser)
    _add_index_overrides(rebuild_parser)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed notes")
    research_parser = subparsers.add_parser("research", help="Run a visible multi-step research workflow")
    ask_parser.add_argument("question", help="Question to ask about the indexed vault")
    research_parser.add_argument("question", help="Research goal to decompose and investigate")
    _add_query_arguments(ask_parser)
    _add_query_arguments(research_parser)
    research_parser.add_argument(
        "--max-subquestions",
        type=int,
        default=3,
        help="Maximum number of explicit research subquestions to generate.",
    )
    ingest_parser.add_argument("url", help="Webpage URL to ingest into the vault")
    ingest_parser.add_argument("--title", help="Optional title override for the saved note")
    ingest_parser.add_argument("--genre", help="Optional genre folder for this import")
    ingest_parser.add_argument(
        "--index-now",
        action="store_true",
        help="Run incremental indexing immediately after saving the ingested note.",
    )
    ingest_youtube_parser.add_argument("url", help="YouTube URL to ingest into the vault")
    ingest_youtube_parser.add_argument("--title", help="Optional title override for the saved note")
    ingest_youtube_parser.add_argument("--genre", help="Optional genre folder for this import")
    ingest_youtube_parser.add_argument(
        "--index-now",
        action="store_true",
        help="Run incremental indexing immediately after saving the ingested note.",
    )
    ingest_pdf_parser.add_argument("file_path", help="Local PDF file to ingest into the vault")
    ingest_pdf_parser.add_argument("--title", help="Optional title override for the saved note")
    ingest_pdf_parser.add_argument("--genre", help="Optional genre folder for this import")
    ingest_pdf_parser.add_argument(
        "--index-now",
        action="store_true",
        help="Run incremental indexing immediately after saving the ingested note.",
    )
    ingest_docx_parser.add_argument("file_path", help="Local DOCX file to ingest into the vault")
    ingest_docx_parser.add_argument("--title", help="Optional title override for the saved note")
    ingest_docx_parser.add_argument("--genre", help="Optional genre folder for this import")
    ingest_docx_parser.add_argument(
        "--index-now",
        action="store_true",
        help="Run incremental indexing immediately after saving the ingested note.",
    )
    return parser


def run_index(config: AppConfig, *, reset_store: bool) -> None:
    """Index vault notes into the local vector store."""
    IndexService(config).index(reset_store=reset_store)


def run_ask(
    config: AppConfig,
    question: str,
    *,
    folder: str | None = None,
    path_contains: str | None = None,
    tag: str | None = None,
    boost_tags: list[str] | None = None,
    include_linked: bool = False,
    top_k: int | None = None,
    candidate_count: int | None = None,
    rerank: bool = False,
    auto_save: bool = False,
    retrieval_scope: str = "knowledge",
    retrieval_mode: str = "local_only",
    answer_mode: str = "balanced",
    track_id: str | None = None,
    use_track_context: bool = False,
    section_focus: str | None = None,
) -> None:
    """Answer a question from the indexed vault."""
    if use_track_context and not (track_id or "").strip():
        raise ValueError("--use-track-context requires --track-id.")
    if top_k is not None and top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if candidate_count is not None and candidate_count < 1:
        raise ValueError("--candidate-count must be at least 1.")
    if top_k is not None and candidate_count is not None and candidate_count < top_k:
        raise ValueError("--candidate-count must be greater than or equal to --top-k.")

    filters = RetrievalFilters(
        folder=folder.strip().strip("/") if folder else None,
        path_contains=path_contains.strip().lower() if path_contains else None,
        tag=tag.strip().lstrip("#").lower() if tag else None,
    )
    options = RetrievalOptions(
        top_k=top_k,
        candidate_count=candidate_count,
        rerank=True if rerank else None,
        boost_tags=tuple(
            tag_value.strip().lstrip("#").lower()
            for tag_value in (boost_tags or [])
            if tag_value.strip()
        ),
        include_linked_notes=True if include_linked else None,
    )
    query_service = QueryService(
        config,
        retriever_cls=Retriever,
        web_search_service_cls=WebSearchService,
        capture_debug_trace=False,
    )
    request = QueryRequest(
        question=question,
        filters=filters,
        options=options,
        auto_save=False,
        retrieval_scope=retrieval_scope,
        retrieval_mode=retrieval_mode,
        answer_mode=answer_mode,
        track_id=track_id.strip() if track_id else None,
        use_track_context=use_track_context,
        section_focus=section_focus.strip() if section_focus else None,
    )

    response = query_service.ask(request)

    if not response.retrieved_chunks and not response.web_results:
        raise RuntimeError("No indexed notes or web results matched the requested retrieval mode and filters.")

    print("\nAnswer\n------")
    print(response.answer)

    print("\nSources")
    print("-------")
    if response.sources:
        for source in response.sources:
            print(f"- {source}")
    else:
        print("- No sources retrieved")

    for warning in response.warnings:
        print(f"\nWarning: {warning}")

    _maybe_review_track_context_update(
        query_service,
        response,
        explicit_track_id=track_id,
    )

    if auto_save or config.auto_save_answer:
        saved_path = save_answer(
            config.draft_answers_path,
            question,
            response.answer_result,
            source_type="saved_answer",
            status="draft",
            indexed=False,
            track_context=response.track_context,
            track_context_update=response.track_context_update,
            active_section_focus=response.debug.active_section or section_focus,
        )
        logger.info("Saved answer to %s", saved_path)
    elif prompt_to_save():
        saved_response = query_service.save(
            question,
            response.answer_result,
            existing_response=response,
        )
        logger.info("Saved answer to %s", saved_response.saved_path)


def run_research(
    config: AppConfig,
    question: str,
    *,
    folder: str | None = None,
    path_contains: str | None = None,
    tag: str | None = None,
    boost_tags: list[str] | None = None,
    include_linked: bool = False,
    top_k: int | None = None,
    candidate_count: int | None = None,
    rerank: bool = False,
    auto_save: bool = False,
    retrieval_scope: str = "knowledge",
    retrieval_mode: str = "local_only",
    answer_mode: str = "balanced",
    max_subquestions: int = 3,
) -> None:
    """Run a visible multi-step research workflow."""
    if max_subquestions < 1:
        raise ValueError("--max-subquestions must be at least 1.")
    filters = RetrievalFilters(
        folder=folder.strip().strip("/") if folder else None,
        path_contains=path_contains.strip().lower() if path_contains else None,
        tag=tag.strip().lstrip("#").lower() if tag else None,
    )
    options = RetrievalOptions(
        top_k=top_k,
        candidate_count=candidate_count,
        rerank=True if rerank else None,
        boost_tags=tuple(
            tag_value.strip().lstrip("#").lower()
            for tag_value in (boost_tags or [])
            if tag_value.strip()
        ),
        include_linked_notes=True if include_linked else None,
    )
    response = ResearchService(config).research(
        ResearchRequest(
            goal=question,
            filters=filters,
            options=options,
            retrieval_scope=retrieval_scope,
            retrieval_mode=retrieval_mode,
            answer_mode=answer_mode,
            max_subquestions=max_subquestions,
            auto_save=False,
        )
    )

    print("\nResearch Plan\n-------------")
    for index, subquestion in enumerate(response.subquestions, start=1):
        print(f"{index}. {subquestion}")

    print("\nFindings\n--------")
    for index, step in enumerate(response.steps, start=1):
        print(f"[Step {index}] {step.subquestion}")
        print(step.response.answer)
        if step.response.sources:
            print("Sources:")
            for source in step.response.sources:
                print(f"- {source}")
        if step.response.warnings:
            for warning in step.response.warnings:
                print(f"Warning: {warning}")
        print("")

    print("Final Research Answer\n---------------------")
    print(response.answer)

    print("\nSources")
    print("-------")
    if response.sources:
        for source in response.sources:
            print(f"- {source}")
    else:
        print("- No sources retrieved")

    for warning in response.warnings:
        print(f"\nWarning: {warning}")

    if auto_save or config.auto_save_answer:
        saved_response = ResearchService(config).save(question, response.answer_result, existing_response=response)
        logger.info("Saved research answer to %s", saved_response.saved_path)
    elif prompt_to_save():
        saved_response = ResearchService(config).save(question, response.answer_result, existing_response=response)
        logger.info("Saved research answer to %s", saved_response.saved_path)


def run_ingest_webpage(
    config: AppConfig,
    url: str,
    *,
    title: str | None = None,
    index_now: bool = False,
    genre: str | None = None,
) -> None:
    """Import a webpage into the vault and optionally index it."""
    response = IngestionService(config).ingest_webpage(
        IngestionRequest(
            source=url,
            title_override=title,
            index_now=True if index_now else None,
            import_genre=genre,
        )
    )

    print("\nIngestion Complete\n------------------")
    print(f"Title: {response.title}")
    print(f"Saved Path: {response.saved_path}")
    print(f"Source Type: {response.source_type}")
    print(f"Indexed Now: {'yes' if response.index_triggered else 'no'}")
    for warning in response.warnings:
        print(f"\nWarning: {warning}")


def run_ingest_youtube(
    config: AppConfig,
    url: str,
    *,
    title: str | None = None,
    index_now: bool = False,
    genre: str | None = None,
) -> None:
    """Import a YouTube transcript into the vault and optionally index it."""
    response = IngestionService(config).ingest_youtube(
        IngestionRequest(
            source=url,
            title_override=title,
            index_now=True if index_now else None,
            import_genre=genre,
        )
    )

    print("\nIngestion Complete\n------------------")
    print(f"Title: {response.title}")
    print(f"Saved Path: {response.saved_path}")
    print(f"Source Type: {response.source_type}")
    print(f"Indexed Now: {'yes' if response.index_triggered else 'no'}")
    for warning in response.warnings:
        print(f"\nWarning: {warning}")


def run_ingest_pdf(
    config: AppConfig,
    file_path: str,
    *,
    title: str | None = None,
    index_now: bool = False,
    genre: str | None = None,
) -> None:
    """Import a PDF into the vault and optionally index it."""
    response = IngestionService(config).ingest_pdf(
        IngestionRequest(
            source=file_path,
            title_override=title,
            index_now=True if index_now else None,
            import_genre=genre,
        )
    )

    print("\nIngestion Complete\n------------------")
    print(f"Title: {response.title}")
    print(f"Saved Path: {response.saved_path}")
    print(f"Source Type: {response.source_type}")
    print(f"Indexed Now: {'yes' if response.index_triggered else 'no'}")
    for warning in response.warnings:
        print(f"\nWarning: {warning}")


def run_ingest_docx(
    config: AppConfig,
    file_path: str,
    *,
    title: str | None = None,
    index_now: bool = False,
    genre: str | None = None,
) -> None:
    """Import a DOCX file into the vault and optionally index it."""
    response = IngestionService(config).ingest_docx(
        IngestionRequest(
            source=file_path,
            title_override=title,
            index_now=True if index_now else None,
            import_genre=genre,
        )
    )

    print("\nIngestion Complete\n------------------")
    print(f"Title: {response.title}")
    print(f"Saved Path: {response.saved_path}")
    print(f"Source Type: {response.source_type}")
    print(f"Indexed Now: {'yes' if response.index_triggered else 'no'}")
    for warning in response.warnings:
        print(f"\nWarning: {warning}")


def _add_index_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override the configured chunk size for this indexing run",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Override the configured chunk overlap for this indexing run",
    )
    parser.add_argument(
        "--chunking-strategy",
        choices=["markdown", "sentence"],
        help="Override the chunking strategy for this indexing run",
    )


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--folder",
        help="Only retrieve notes from this vault-relative folder",
    )
    parser.add_argument(
        "--path-contains",
        help="Only retrieve notes whose path contains this text",
    )
    parser.add_argument(
        "--tag",
        help="Only retrieve notes that contain this tag",
    )
    parser.add_argument(
        "--boost-tag",
        action="append",
        default=[],
        help="Boost notes matching this tag during retrieval. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-linked",
        action="store_true",
        help="Include context from notes linked by the primary retrieved notes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override the number of final chunks used to answer the question",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        help="Override the number of chunks retrieved before reranking",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable simple heuristic reranking for this query",
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Save the generated answer without prompting.",
    )
    parser.add_argument(
        "--retrieval-scope",
        choices=[scope.value for scope in RetrievalScope],
        default=RetrievalScope.KNOWLEDGE.value,
        help="Choose curated knowledge only, or a broader extended local scope that may include lower-trust drafts, research outputs, and imports.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.LOCAL_ONLY.value,
        help="Choose whether to use only local notes, automatic web fallback, or hybrid local+web retrieval.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=[mode.value for mode in AnswerMode],
        default=AnswerMode.BALANCED.value,
        help="Choose strict evidence-bound answers, balanced evidence-first answers, or exploratory synthesis.",
    )
    parser.add_argument(
        "--track-id",
        help="Active in-progress track ID for YAML Track Context memory.",
    )
    parser.add_argument(
        "--use-track-context",
        action="store_true",
        help="Load YAML Track Context for the provided --track-id before answering.",
    )
    parser.add_argument(
        "--section-focus",
        help="Optional active section focus to carry into the current turn, such as drop or breakdown.",
    )


def _config_with_index_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    chunk_size = args.chunk_size if getattr(args, "chunk_size", None) is not None else config.chunk_size
    chunk_overlap = (
        args.chunk_overlap if getattr(args, "chunk_overlap", None) is not None else config.chunk_overlap
    )
    chunking_strategy = (
        args.chunking_strategy
        if getattr(args, "chunking_strategy", None) is not None
        else config.chunking_strategy
    )

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size.")

    return replace(
        config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
    )


def _maybe_review_track_context_update(
    query_service: QueryService,
    response: QueryResponse,
    *,
    explicit_track_id: str | None,
) -> None:
    proposal = response.track_context_update
    active_track_context = response.track_context
    if proposal is None or active_track_context is None or proposal.is_empty():
        return

    print("\nSuggested Track Context Update")
    print("-----------------------------")
    if proposal.summary.strip():
        print(f"Summary: {proposal.summary.strip()}")
    if proposal.confidence.strip():
        print(f"Confidence: {proposal.confidence.strip()}")
    if proposal.source_reasoning.strip():
        print(f"Why suggested: {proposal.source_reasoning.strip()}")
    for heading, items in proposal_groups(proposal):
        print(f"\n{heading}")
        for item in items:
            print(f"- {item}")

    preview_context = query_service.track_context_update_service.preview(
        active_track_context,
        proposal,
    )
    print("\nUpdated Track Context Preview")
    print("----------------------------")
    print(f"Track ID: {preview_context.track_id}")
    if preview_context.track_name:
        print(f"Title: {preview_context.track_name}")
    if preview_context.genre:
        print(f"Genre: {preview_context.genre}")
    if preview_context.current_stage:
        print(f"Current Stage: {preview_context.current_stage}")
    if preview_context.current_problem:
        print(f"Current Problem: {preview_context.current_problem}")
    if preview_context.known_issues:
        print(f"Known Issues: {', '.join(preview_context.known_issues)}")
    if preview_context.goals:
        print(f"Goals: {', '.join(preview_context.goals)}")
    if preview_context.sections:
        print("Sections:")
        for section_key, section in preview_context.sections.items():
            summary_parts = [section.name or section_key]
            if section.role:
                summary_parts.append(f"role={section.role}")
            if section.energy_level:
                summary_parts.append(f"energy={section.energy_level}")
            if section.bars:
                summary_parts.append(f"bars={section.bars}")
            if section.issues:
                summary_parts.append(f"issues={', '.join(section.issues)}")
            print(f"- {section_key}: {' | '.join(summary_parts)}")

    if proposal.section_focus.strip():
        print(
            f"\nSession note: section focus would move to `{proposal.section_focus.strip()}` for the next turn."
        )

    target_track_id = explicit_track_id.strip() if explicit_track_id else active_track_context.track_id
    if not target_track_id:
        return
    if not _prompt_yes_no("\nApply this Track Context update to the YAML file now? (y/n): "):
        return

    updated_context = query_service.track_context_update_service.apply(active_track_context, proposal)
    saved_path = query_service.track_context_service.save(updated_context)
    print(f"Saved updated Track Context to {saved_path}")


def _prompt_yes_no(prompt_text: str) -> bool:
    response = input(prompt_text).strip().lower()
    return response in {"y", "yes"}


def _resolve_note_links(notes: list[Note]) -> None:
    """Compatibility wrapper for older tests and call sites."""
    resolve_note_links(notes)


def _build_note_alias_map(notes: list[Note]) -> dict[str, str]:
    """Compatibility wrapper for older tests and call sites."""
    return build_note_alias_map(notes)


def _ensure_index_compatible(vector_store) -> None:
    """Compatibility wrapper for older tests and call sites."""
    ensure_index_compatible(vector_store)


if __name__ == "__main__":
    sys.exit(main())
