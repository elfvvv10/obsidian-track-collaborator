"""CLI entrypoint for the local Obsidian RAG assistant."""

from __future__ import annotations

import argparse
import sys

from agent import ResearchAgent
from chunker import chunk_notes
from config import AppConfig, load_config
from embeddings import OllamaEmbeddingClient
from llm import OllamaChatClient
from retriever import Retriever
from saver import prompt_to_save, save_answer
from utils import RetrievalFilters, get_logger
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
            run_index(config, reset_store=False)
        elif args.command == "rebuild":
            run_index(config, reset_store=True)
        elif args.command == "ask":
            run_ask(
                config,
                args.question,
                folder=args.folder,
                path_contains=args.path_contains,
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
    subparsers.add_parser("rebuild", help="Clear and rebuild the vector index")

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
    return parser


def run_index(config: AppConfig, *, reset_store: bool) -> None:
    """Index vault notes into the local vector store."""
    logger.info("Loading notes from %s", config.obsidian_vault_path)
    notes = load_notes(config.obsidian_vault_path)
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

    chunks = chunk_notes(notes)
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
) -> None:
    """Answer a question from the indexed vault."""
    embedding_client = OllamaEmbeddingClient(config)
    vector_store = VectorStore(config)
    retriever = Retriever(config, embedding_client, vector_store)
    chat_client = OllamaChatClient(config)
    agent = ResearchAgent(retriever, chat_client)
    filters = RetrievalFilters(
        folder=folder.strip().strip("/") if folder else None,
        path_contains=path_contains.strip().lower() if path_contains else None,
    )

    logger.info("Retrieving relevant notes")
    result = agent.answer(question, filters=filters)

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

    if prompt_to_save():
        saved_path = save_answer(config.obsidian_output_path, question, result)
        logger.info("Saved answer to %s", saved_path)


def _group_chunks_by_note_key(chunks: list) -> dict[str, list]:
    grouped_chunks: dict[str, list] = {}
    for chunk in chunks:
        grouped_chunks.setdefault(chunk.note_key, []).append(chunk)
    return grouped_chunks


if __name__ == "__main__":
    sys.exit(main())
