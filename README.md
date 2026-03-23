# Obsidian RAG Assistant

A local-first Python Obsidian RAG assistant that runs through a CLI or a lightweight local Streamlit UI, using Ollama for inference and ChromaDB for vector search.

## Features

- Reads Markdown notes from an Obsidian vault
- Chunks note content into retrieval-friendly Markdown-aware or sentence-aware segments
- Creates embeddings locally with Ollama
- Stores embeddings in a local ChromaDB database
- Retrieves relevant note chunks for a question
- Supports optional retrieval filters by folder and path text
- Supports configurable chunk sizing, candidate retrieval depth, and optional reranking
- Parses frontmatter and stores tags for filtering and boosting
- Detects Obsidian note links and can optionally include linked-note context
- Supports an improved saved-note template and optional auto-save
- Generates grounded answers with a local Ollama chat model
- Shows source note references in the terminal
- Includes a lightweight local Streamlit UI for asking questions, indexing, and debugging
- Optionally saves answers back into the vault as Markdown notes
- Uses incremental indexing to update only changed notes
- Excludes saved answers in the configured output folder from indexing when that folder lives inside the vault
- Includes mocked tests, local smoke tests, and phase-focused module tests
- Includes a sample vault for quick testing

## How It Works

Retrieval-augmented generation, or RAG, is a simple pattern:

1. Your notes are split into chunks.
2. Each chunk is converted into an embedding, which is a numeric representation of meaning.
3. Those embeddings are stored in a vector database.
4. When you ask a question, the app embeds the question and finds the most relevant note chunks.
5. The retrieved chunks are sent to the language model along with your question.
6. The model answers using that note context instead of relying only on its own memory.

This helps keep answers grounded in your own notes.

## Architecture Overview

The current flow is:

`Obsidian vault -> vault loader -> configurable chunker -> Ollama embeddings -> ChromaDB -> retriever -> optional reranker -> Ollama chat -> terminal answer -> optional save-back`

The app now also includes a thin service layer so both the CLI and UI can share the same orchestration path without duplicating business logic.

Core modules:

- `main.py`: CLI entrypoint
- `config.py`: environment loading and validation
- `services/index_service.py`: shared indexing/build flow for CLI and UI
- `services/query_service.py`: shared query + answer flow for CLI and UI
- `services/models.py`: structured request/response models for service consumers
- `services/common.py`: shared service helpers such as link resolution and index checks
- `streamlit_app.py`: lightweight local UI
- `vault_loader.py`: Markdown vault scanning
- `chunker.py`: configurable Markdown-aware and sentence-aware chunk creation
- `embeddings.py`: Ollama embedding API client
- `llm.py`: Ollama chat API client
- `vector_store.py`: ChromaDB persistence
- `retriever.py`: query embedding + candidate retrieval + optional reranking
- `reranker.py`: lightweight heuristic reranking
- `metadata_parser.py`: frontmatter and tag parsing helpers
- `link_parser.py`: Obsidian link parsing helpers
- `agent.py`: retrieval + answer orchestration
- `saver.py`: save answer back to Markdown
- `utils.py`: shared models and helpers

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/)
- Local Ollama models:
  - `hermes3`
  - `nomic-embed-text`

## Installation

Clone the repo and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama Setup

### 1. Install Ollama

Follow the official instructions for your platform:

- macOS / Linux / Windows: [https://ollama.com/download](https://ollama.com/download)

### 2. Pull the required models

```bash
ollama pull hermes3
ollama pull nomic-embed-text
```

### 3. Verify Ollama is running

Start Ollama if needed, then verify the local API:

```bash
curl http://localhost:11434/api/tags
```

You should get a JSON response listing installed models.

## Environment Setup

Copy the example environment file:

```bash
cp .env.example .env
```

Example `.env`:

```env
OBSIDIAN_VAULT_PATH=./sample_vault
OBSIDIAN_OUTPUT_PATH=./sample_vault/research_answers
CHROMA_DB_PATH=./chroma_db
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=hermes3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
TOP_K_RESULTS=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_CANDIDATE_MULTIPLIER=2
CHUNKING_STRATEGY=markdown
ENABLE_RERANKING=false
TAG_BOOST_WEIGHT=3.0
ENABLE_LINKED_NOTE_EXPANSION=false
MAX_LINKED_NOTES=2
LINKED_NOTE_CHUNKS_PER_NOTE=1
AUTO_SAVE_ANSWER=false
```

Variable notes:

- `OBSIDIAN_VAULT_PATH`: path to the vault you want to index
- `OBSIDIAN_OUTPUT_PATH`: where saved answer notes should be written
- `CHROMA_DB_PATH`: directory for local ChromaDB storage
- `OLLAMA_BASE_URL`: Ollama HTTP API base URL
- `OLLAMA_CHAT_MODEL`: local chat model
- `OLLAMA_EMBEDDING_MODEL`: local embedding model
- `TOP_K_RESULTS`: number of chunks to retrieve per question
- `CHUNK_SIZE`: default chunk size used during indexing
- `CHUNK_OVERLAP`: overlap between adjacent chunks
- `RETRIEVAL_CANDIDATE_MULTIPLIER`: fetch more candidates before final selection
- `CHUNKING_STRATEGY`: `markdown` or `sentence`
- `ENABLE_RERANKING`: enable simple heuristic reranking by default
- `TAG_BOOST_WEIGHT`: extra ranking weight for chunks whose tags match `--boost-tag`
- `ENABLE_LINKED_NOTE_EXPANSION`: include linked note context by default
- `MAX_LINKED_NOTES`: maximum linked notes to expand per question
- `LINKED_NOTE_CHUNKS_PER_NOTE`: chunks to include from each linked note
- `AUTO_SAVE_ANSWER`: save answers automatically without prompting

## Index Your Notes

Build the local vector index:

```bash
python main.py index
```

Optional indexing overrides:

```bash
python main.py index --chunk-size 800 --chunk-overlap 100
python main.py index --chunking-strategy sentence
```

`index` is incremental. It updates changed notes, adds new notes, and removes deleted notes without rebuilding the whole collection.

Rebuild from scratch:

```bash
python main.py rebuild
```

## Ask Questions

Once your notes are indexed:

```bash
python main.py ask "What do my notes say about AI agents?"
```

Optional filters:

```bash
python main.py ask "What do my notes say about AI agents?" --folder projects
python main.py ask "What do my notes say about AI agents?" --path-contains agents
python main.py ask "What do my notes say about AI agents?" --tag ai
python main.py ask "What do my notes say about AI agents?" --boost-tag agents --boost-tag local-ai
python main.py ask "What do my notes say about AI agents?" --include-linked
python main.py ask "What do my notes say about AI agents?" --auto-save
python main.py ask "What do my notes say about AI agents?" --top-k 2 --candidate-count 6 --rerank
```

The app will:

1. Embed your question locally
2. Retrieve the top matching note chunks from ChromaDB
3. Send the retrieved context and your question to Ollama
4. Print the answer and sources in the terminal
5. Ask whether you want to save the answer as a Markdown note

## Run the Local UI

The Streamlit UI uses the same service layer as the CLI, so indexing and question-answering behavior stay aligned.

Start the UI with:

```bash
streamlit run streamlit_app.py
```

The UI includes three sections:

- `Ask`: question input, answer display, sources, linked-note context, and an optional debug view of retrieved chunks
- `Index`: build and rebuild actions with user-friendly status messages
- `Settings / Debug`: active models, top-k control, reranking toggle, linked-note toggle, auto-save toggle, and retrieval filters

The UI is local-only and does not add any cloud services, authentication, or background job system.

## Save-Back Behavior

After each answer, the CLI prompts:

```text
Save this answer to your Obsidian output folder? (y/n):
```

If you answer `y`, or if auto-save is enabled, the app creates a Markdown note containing:

- The original question
- A short summary
- The full answer
- Key points
- Sources used

If you save the same question repeatedly, the app keeps existing notes and creates deterministic suffixes such as `-answer-2.md`, `-answer-3.md`, and so on.

## Example Workflow

```bash
cp .env.example .env
python main.py index
python main.py ask "What themes recur in my product notes?"
```

If you keep the sample settings, saved answers will appear in:

```text
sample_vault/research_answers/
```

When `OBSIDIAN_OUTPUT_PATH` points to a folder inside the vault, saved answer notes in that folder are excluded from indexing by default so they do not pollute retrieval.

## Project Structure

```text
.
├── main.py
├── config.py
├── vault_loader.py
├── chunker.py
├── embeddings.py
├── llm.py
├── metadata_parser.py
├── link_parser.py
├── vector_store.py
├── retriever.py
├── reranker.py
├── agent.py
├── streamlit_app.py
├── saver.py
├── utils.py
├── services/
│   ├── common.py
│   ├── index_service.py
│   ├── models.py
│   └── query_service.py
├── requirements.txt
├── .env.example
├── README.md
├── sample_vault/
└── tests/
```

The `tests/` directory includes:

- mocked client and CLI tests
- local module and smoke tests
- orchestration-level integration tests using temporary vaults and real local indexing/retrieval flow
- phase-focused tests for retrieval, metadata, links, and save-back behavior

## Troubleshooting

### Ollama is not running

Symptom:

- You see an error saying Ollama could not be reached.

Fix:

- Start Ollama
- Verify `OLLAMA_BASE_URL`
- Run:

```bash
curl http://localhost:11434/api/tags
```

### Required model is missing

Symptom:

- The app says `hermes3` or `nomic-embed-text` is not installed.

Fix:

```bash
ollama pull hermes3
ollama pull nomic-embed-text
```

### No notes were indexed

Symptom:

- The app says no Markdown notes were found.

Fix:

- Confirm `OBSIDIAN_VAULT_PATH` points to a real vault directory
- Make sure the vault contains `.md` files
- Hidden folders and `.obsidian/` are intentionally ignored

### Retrieval returns weak answers

Symptom:

- The answer says context is missing or insufficient.

Fix:

- Rebuild the index after changing notes
- Increase note coverage in your vault
- Raise `TOP_K_RESULTS`
- Improve note clarity or add more descriptive headings

### Retrieval filters return no results

Symptom:

- The app says no indexed notes matched the requested retrieval filters.

Fix:

- Check the folder path you passed to `--folder`
- Check the substring you passed to `--path-contains`
- Retry without filters to confirm the note is indexed
- Run `python main.py index` after adding or moving notes

### Index format is out of date

Symptom:

- The app tells you the local index format is out of date and asks you to rebuild.

Fix:

```bash
python main.py rebuild
```

This can happen after retrieval-relevant schema changes such as new metadata fields or index layout updates.

## Current Limitations

- Chunking is Markdown-aware but still heuristic rather than token-aware
- Metadata filters are still intentionally simple: folder, path text, and tag-based controls only
- The Streamlit UI is intentionally lightweight and does not yet include persistent chat history or advanced source inspection workflows
- Automated tests are strong locally, but live Ollama behavior is still mostly verified manually
- Prompting is intentionally simple

## Suggested V2 Roadmap

- Source snippet highlighting
- Configurable prompt templates
- Metadata filtering by tags, frontmatter, or note type
- Conversation history
- Richer UI features such as persistent sessions and better source inspection
- Optional live integration checks for Ollama and Chroma
