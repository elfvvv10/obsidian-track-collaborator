# Obsidian RAG Assistant

A local-first Python CLI that turns an Obsidian vault into a retrieval-augmented research assistant using Ollama for inference and ChromaDB for vector search.

## Features

- Reads Markdown notes from an Obsidian vault
- Chunks note content into retrieval-friendly Markdown-aware segments
- Creates embeddings locally with Ollama
- Stores embeddings in a local ChromaDB database
- Retrieves relevant note chunks for a question
- Supports optional retrieval filters by folder and path text
- Generates grounded answers with a local Ollama chat model
- Shows source note references in the terminal
- Optionally saves answers back into the vault as Markdown notes
- Uses incremental indexing to update only changed notes
- Includes mocked test coverage for core clients and CLI behavior
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

The v1 flow is:

`Obsidian vault -> vault loader -> Markdown-aware chunker -> Ollama embeddings -> ChromaDB -> retriever -> Ollama chat -> terminal answer -> optional save-back`

Core modules:

- `main.py`: CLI entrypoint
- `config.py`: environment loading and validation
- `vault_loader.py`: Markdown vault scanning
- `chunker.py`: Markdown-aware chunk creation with overlap
- `embeddings.py`: Ollama embedding API client
- `llm.py`: Ollama chat API client
- `vector_store.py`: ChromaDB persistence
- `retriever.py`: query embedding + similarity lookup with optional filters
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
```

Variable notes:

- `OBSIDIAN_VAULT_PATH`: path to the vault you want to index
- `OBSIDIAN_OUTPUT_PATH`: where saved answer notes should be written
- `CHROMA_DB_PATH`: directory for local ChromaDB storage
- `OLLAMA_BASE_URL`: Ollama HTTP API base URL
- `OLLAMA_CHAT_MODEL`: local chat model
- `OLLAMA_EMBEDDING_MODEL`: local embedding model
- `TOP_K_RESULTS`: number of chunks to retrieve per question

## Index Your Notes

Build the local vector index:

```bash
python main.py index
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
```

The app will:

1. Embed your question locally
2. Retrieve the top matching note chunks from ChromaDB
3. Send the retrieved context and your question to Ollama
4. Print the answer and sources in the terminal
5. Ask whether you want to save the answer as a Markdown note

## Save-Back Behavior

After each answer, the CLI prompts:

```text
Save this answer to your Obsidian output folder? (y/n):
```

If you answer `y`, the app creates a Markdown note containing:

- A title
- A timestamp
- The original question
- The generated answer
- The sources used

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

## Project Structure

```text
.
├── main.py
├── config.py
├── vault_loader.py
├── chunker.py
├── embeddings.py
├── llm.py
├── vector_store.py
├── retriever.py
├── agent.py
├── saver.py
├── utils.py
├── requirements.txt
├── .env.example
├── README.md
├── sample_vault/
└── tests/
```

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

## Limitations of V1

- Chunking is Markdown-aware but still heuristic rather than token-aware
- Metadata filters currently support folder and path text only
- No GUI or web app
- Automated tests use mocks rather than live Ollama integrations
- Prompting is intentionally simple

## Suggested V2 Roadmap

- Source snippet highlighting
- Configurable prompt templates
- Metadata filtering by tags, frontmatter, or note type
- Conversation history
- Simple TUI or desktop UI
- Optional live integration checks for Ollama and Chroma
