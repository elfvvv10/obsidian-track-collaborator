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
from utils import get_logger
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
            run_index(config, reset_store=True)
        elif args.command == "rebuild":
            run_index(config, reset_store=True)
        elif args.command == "ask":
            run_ask(config, args.question)
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
    return parser


def run_index(config: AppConfig, *, reset_store: bool) -> None:
    """Index vault notes into the local vector store."""
    logger.info("Loading notes from %s", config.obsidian_vault_path)
    notes = load_notes(config.obsidian_vault_path)
    if not notes:
        raise RuntimeError("No markdown notes were found in the configured vault.")

    chunks = chunk_notes(notes)
    if not chunks:
        raise RuntimeError("No note chunks were created from the vault contents.")

    logger.info("Loaded %s notes and created %s chunks", len(notes), len(chunks))

    embedding_client = OllamaEmbeddingClient(config)
    vector_store = VectorStore(config)
    if reset_store:
        logger.info("Resetting Chroma collection")
        vector_store.reset()

    logger.info("Generating embeddings with Ollama model '%s'", config.ollama_embedding_model)
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])

    logger.info("Writing chunks to ChromaDB at %s", config.chroma_db_path)
    vector_store.upsert_chunks(chunks, embeddings)
    logger.info("Index complete. Stored %s chunks.", vector_store.count())


def run_ask(config: AppConfig, question: str) -> None:
    """Answer a question from the indexed vault."""
    embedding_client = OllamaEmbeddingClient(config)
    vector_store = VectorStore(config)
    retriever = Retriever(config, embedding_client, vector_store)
    chat_client = OllamaChatClient(config)
    agent = ResearchAgent(retriever, chat_client)

    logger.info("Retrieving relevant notes")
    result = agent.answer(question)

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


if __name__ == "__main__":
    sys.exit(main())
