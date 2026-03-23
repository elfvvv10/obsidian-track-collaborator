"""CLI entrypoint for the local Obsidian RAG assistant."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys

from agent import ResearchAgent
from chunker import chunk_notes
from config import AppConfig, load_config
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from retriever import Retriever
from saver import prompt_to_save, save_answer
from utils import RetrievalFilters, RetrievalOptions, get_logger, make_note_key, normalize_path
from vault_loader import load_notes
from vector_store import VectorStore


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

    index_parser = subparsers.choices["index"]
    _add_index_overrides(index_parser)
    _add_index_overrides(rebuild_parser)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed notes")
    ask_parser.add_argument("question", help="Question to ask about the indexed vault")
    ask_parser.add_argument(
        "--folder",
        help="Only retrieve notes from this vault-relative folder",
    )
    ask_parser.add_argument(
        "--path-contains",
        help="Only retrieve notes whose path contains this text",
    )
    ask_parser.add_argument(
        "--tag",
        help="Only retrieve notes that contain this tag",
    )
    ask_parser.add_argument(
        "--boost-tag",
        action="append",
        default=[],
        help="Boost notes matching this tag during retrieval. Can be passed multiple times.",
    )
    ask_parser.add_argument(
        "--include-linked",
        action="store_true",
        help="Include context from notes linked by the primary retrieved notes.",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        help="Override the number of final chunks used to answer the question",
    )
    ask_parser.add_argument(
        "--candidate-count",
        type=int,
        help="Override the number of chunks retrieved before reranking",
    )
    ask_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable simple heuristic reranking for this query",
    )
    ask_parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Save the generated answer without prompting.",
    )
    return parser


def run_index(config: AppConfig, *, reset_store: bool) -> None:
    """Index vault notes into the local vector store."""
    logger.info("Loading notes from %s", config.obsidian_vault_path)
    excluded_paths = []
    if config.obsidian_output_path != config.obsidian_vault_path:
        excluded_paths.append(config.obsidian_output_path)

    notes = load_notes(config.obsidian_vault_path, excluded_paths=excluded_paths)
    _resolve_note_links(notes)
    embedding_client = OllamaEmbeddingClient(config)
    vector_store = VectorStore(config)
    if reset_store:
        logger.info("Resetting Chroma collection")
        vector_store.reset()

    existing_fingerprints = {} if reset_store else vector_store.list_note_fingerprints()
    if not notes:
        if existing_fingerprints:
            logger.info("Vault is empty. Removing %s indexed note(s).", len(existing_fingerprints))
            vector_store.delete_by_note_keys(list(existing_fingerprints))
            logger.info("Index complete. Stored %s chunks.", vector_store.count())
            return
        raise RuntimeError("No markdown notes were found in the configured vault.")

    chunks = chunk_notes(
        notes,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
        strategy=config.chunking_strategy,
    )
    if not chunks:
        raise RuntimeError("No note chunks were created from the vault contents.")

    logger.info("Loaded %s notes and created %s chunks", len(notes), len(chunks))

    note_chunks = _group_chunks_by_note_key(chunks)

    deleted_note_keys = sorted(set(existing_fingerprints) - set(note_chunks))
    if deleted_note_keys:
        logger.info("Removing %s deleted note(s) from the index", len(deleted_note_keys))
        vector_store.delete_by_note_keys(deleted_note_keys)

    chunks_to_index = []
    for note_key, note_chunk_group in note_chunks.items():
        current_fingerprint = note_chunk_group[0].note_fingerprint
        existing_fingerprint = existing_fingerprints.get(note_key)
        if current_fingerprint == existing_fingerprint:
            continue

        if existing_fingerprint:
            vector_store.delete_by_note_keys([note_key])
        chunks_to_index.extend(note_chunk_group)

    if not chunks_to_index:
        logger.info("Index already up to date. Stored %s chunks.", vector_store.count())
        return

    logger.info(
        "Generating embeddings for %s updated chunk(s) with Ollama model '%s'",
        len(chunks_to_index),
        config.ollama_embedding_model,
    )
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks_to_index])

    logger.info("Writing updated chunks to ChromaDB at %s", config.chroma_db_path)
    vector_store.upsert_chunks(chunks_to_index, embeddings)
    logger.info("Index complete. Stored %s chunks.", vector_store.count())


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
) -> None:
    """Answer a question from the indexed vault."""
    if top_k is not None and top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if candidate_count is not None and candidate_count < 1:
        raise ValueError("--candidate-count must be at least 1.")
    if top_k is not None and candidate_count is not None and candidate_count < top_k:
        raise ValueError("--candidate-count must be greater than or equal to --top-k.")

    embedding_client = OllamaEmbeddingClient(config)
    vector_store = VectorStore(config)
    retriever = Retriever(config, embedding_client, vector_store)
    chat_client = OllamaChatClient(config)
    agent = ResearchAgent(retriever, chat_client)
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

    logger.info("Retrieving relevant notes")
    result = agent.answer(question, filters=filters, options=options)

    if not result.retrieved_chunks:
        raise RuntimeError("No indexed notes matched the requested retrieval filters.")

    print("\nAnswer\n------")
    print(result.answer)

    print("\nSources")
    print("-------")
    if result.sources:
        for source in result.sources:
            print(f"- {source}")
    else:
        print("- No sources retrieved")

    should_save = auto_save or config.auto_save_answer or prompt_to_save()
    if should_save:
        saved_path = save_answer(config.obsidian_output_path, question, result)
        logger.info("Saved answer to %s", saved_path)


def _group_chunks_by_note_key(chunks: list) -> dict[str, list]:
    grouped_chunks: dict[str, list] = {}
    for chunk in chunks:
        grouped_chunks.setdefault(chunk.note_key, []).append(chunk)
    return grouped_chunks


def _resolve_note_links(notes: list) -> None:
    alias_map = _build_note_alias_map(notes)

    for note in notes:
        own_note_key = make_note_key(note.path)
        resolved_keys: list[str] = []
        seen: set[str] = set()
        for link in note.links:
            note_key = alias_map.get(link)
            if not note_key or note_key == own_note_key or note_key in seen:
                continue
            seen.add(note_key)
            resolved_keys.append(note_key)
        note.linked_note_keys = tuple(resolved_keys)


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


def _build_note_alias_map(notes: list) -> dict[str, str]:
    alias_map: dict[str, str] = {}

    for note in notes:
        note_key = make_note_key(note.path)
        normalized_path = normalize_path(note.path).lower()
        aliases = {
            normalized_path,
            normalized_path.rsplit(".", 1)[0],
            normalized_path.split("/")[-1],
            normalized_path.split("/")[-1].rsplit(".", 1)[0],
            note.title.strip().lower(),
        }
        for alias in aliases:
            if alias:
                alias_map[alias] = note_key

    return alias_map


if __name__ == "__main__":
    sys.exit(main())
