"""CLI entrypoint for the local Obsidian RAG assistant."""

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
    ResearchRequest,
    RetrievalMode,
    RetrievalScope,
)
from services.query_service import QueryService
from services.research_service import ResearchService
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
    parser = argparse.ArgumentParser(description="Local Obsidian RAG assistant")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("index", help="Build the vector index from the configured vault")
    rebuild_parser = subparsers.add_parser("rebuild", help="Clear and rebuild the vector index")
    ingest_parser = subparsers.add_parser("ingest-webpage", help="Import a webpage into the vault")
    ingest_youtube_parser = subparsers.add_parser("ingest-youtube", help="Import a YouTube video knowledge note into the vault")

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
    ingest_parser.add_argument(
        "--index-now",
        action="store_true",
        help="Run incremental indexing immediately after saving the ingested note.",
    )
    ingest_youtube_parser.add_argument("url", help="YouTube URL to ingest into the vault")
    ingest_youtube_parser.add_argument("--title", help="Optional title override for the saved note")
    ingest_youtube_parser.add_argument(
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
) -> None:
    """Answer a question from the indexed vault."""
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
        embedding_client_cls=OllamaEmbeddingClient,
        chat_client_cls=OllamaChatClient,
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

    if auto_save or config.auto_save_answer:
        saved_path = save_answer(
            config.draft_answers_path,
            question,
            response.answer_result,
            source_type="saved_answer",
            status="draft",
            indexed=False,
        )
        logger.info("Saved answer to %s", saved_path)
    elif prompt_to_save():
        saved_response = query_service.save(question, response.answer_result)
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
) -> None:
    """Import a webpage into the vault and optionally index it."""
    response = IngestionService(config).ingest_webpage(
        IngestionRequest(
            source=url,
            title_override=title,
            index_now=True if index_now else None,
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
) -> None:
    """Import a YouTube transcript into the vault and optionally index it."""
    response = IngestionService(config).ingest_youtube(
        IngestionRequest(
            source=url,
            title_override=title,
            index_now=True if index_now else None,
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
