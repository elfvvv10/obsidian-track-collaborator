# Obsidian RAG Assistant

A local-first Python Obsidian RAG assistant that runs through a CLI or a lightweight local Streamlit UI, using Ollama for inference and ChromaDB for vector search, with optional external web evidence when enabled.

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
- Supports optional external web search as a separate evidence path
- Supports webpage ingestion as a separate content-import workflow
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

Optional web search is handled separately through the service layer:

`Question -> local retrieval path -> optional web-search service -> Ollama chat with separated local/web evidence`

External content ingestion is handled as a separate workflow:

`Webpage URL -> ingestion service -> webpage fetch/extract -> Markdown note in vault -> optional indexing`

The app now also includes a thin service layer so both the CLI and UI can share the same orchestration path without duplicating business logic.

Core modules:

- `main.py`: CLI entrypoint
- `config.py`: environment loading and validation
- `services/index_service.py`: shared indexing/build flow for CLI and UI
- `services/query_service.py`: shared query + answer flow for CLI and UI
- `services/web_search_service.py`: optional external search orchestration
- `services/ingestion_service.py`: shared external content ingestion orchestration
- `services/models.py`: structured request/response models for service consumers
- `services/common.py`: shared service helpers such as link resolution and index checks
- `services/webpage_ingestion_service.py`: webpage fetch, text extraction, and note creation
- `streamlit_app.py`: lightweight local UI
- `vault_loader.py`: Markdown vault scanning
- `chunker.py`: configurable Markdown-aware and sentence-aware chunk creation
- `embeddings.py`: Ollama embedding API client
- `llm.py`: Ollama chat API client
- `vector_store.py`: ChromaDB persistence
- `retriever.py`: query embedding + candidate retrieval + optional reranking
- `reranker.py`: lightweight heuristic reranking
- `web_search.py`: lightweight web search clients and result model
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
WEB_SEARCH_PROVIDER=wikipedia
WEB_SEARCH_API_URL=
WEB_SEARCH_MAX_RESULTS=3
WEB_SEARCH_TIMEOUT_SECONDS=10
WEBPAGE_INGESTION_FOLDER=ingested_webpages
AUTO_INDEX_AFTER_INGESTION=false
WEBPAGE_FETCH_TIMEOUT_SECONDS=15
WEBPAGE_FETCH_USER_AGENT=obsidian-rag-assistant/1.0
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
- `WEB_SEARCH_PROVIDER`: external search provider. `wikipedia` is the default no-key option. `duckduckgo` remains available as an alternative.
- `WEB_SEARCH_API_URL`: optional provider endpoint override. Leave blank to use the provider default.
- `WEB_SEARCH_MAX_RESULTS`: maximum number of external results to include
- `WEB_SEARCH_TIMEOUT_SECONDS`: timeout for external web search requests
- `WEBPAGE_INGESTION_FOLDER`: vault-relative folder where ingested webpages are saved
- `AUTO_INDEX_AFTER_INGESTION`: automatically run incremental indexing after a successful ingestion
- `WEBPAGE_FETCH_TIMEOUT_SECONDS`: timeout for webpage fetch requests
- `WEBPAGE_FETCH_USER_AGENT`: user-agent string used for webpage ingestion requests

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
python main.py ask "What happened in AI this week?" --retrieval-mode auto
python main.py ask "Summarize local notes and web context for local models" --retrieval-mode hybrid
```

The app will:

1. Embed your question locally
2. Retrieve the top matching note chunks from ChromaDB
3. Send the retrieved context and your question to Ollama
4. Print the answer and sources in the terminal
5. Ask whether you want to save the answer as a Markdown note

### Retrieval Modes

- `local_only`: use only your Obsidian vault. This is the default and preserves the original local-first behavior.
- `auto`: try local retrieval first, and use web search only when local evidence is missing or weak.
- `hybrid`: use local retrieval and web search together, even if local evidence exists.

When web search is used, local note sources and web sources are labeled separately in both the CLI and UI.
The default provider is now Wikipedia search because it is a more reliable no-key option than the previous DuckDuckGo-only path.

## Ingest Webpages

Webpage ingestion is an import workflow, not a query-time retrieval feature. It fetches a page, extracts readable content, saves that content into your vault as Markdown, and can then use the normal indexing pipeline so the page becomes part of your local knowledge base.

Run it from the CLI:

```bash
python main.py ingest-webpage "https://example.com/article"
python main.py ingest-webpage "https://example.com/article" --title "Example Article" --index-now
```

Saved webpage notes go into `WEBPAGE_INGESTION_FOLDER` inside your vault, which defaults to `ingested_webpages/`.

Each saved note includes:

- page title
- source URL
- ingestion timestamp
- extracted readable content

## Run the Local UI

The Streamlit UI uses the same service layer as the CLI, so indexing and question-answering behavior stay aligned.

Start the UI with:

```bash
streamlit run streamlit_app.py
```

The UI includes four main areas:

- `Sidebar`: query filters and retrieval controls such as folder, path text, tag, top-k, reranking, linked-note expansion, auto-save, and retrieval mode
- `Ask`: question input, answer display, separate local/web sources, save actions, linked-note context, and an optional debug view of retrieval stages
- `Ingest`: paste a webpage URL, save it into the vault, and optionally trigger indexing right away
- `Index`: readiness messages plus build and rebuild actions
- `Settings / Debug`: active models, paths, app readiness, index compatibility, and the debug toggle

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

In the Streamlit UI, you can also provide an optional title override before saving. That title is used for the note heading and filename slug, while the underlying save logic stays the same.

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
├── web_search.py
├── agent.py
├── streamlit_app.py
├── saver.py
├── utils.py
├── services/
│   ├── common.py
│   ├── ingestion_service.py
│   ├── index_service.py
│   ├── models.py
│   ├── query_service.py
│   ├── web_search_service.py
│   └── webpage_ingestion_service.py
├── requirements.txt
├── .env.example
├── README.md
├── sample_vault/
└── tests/
```

The `tests/` directory includes:

- mocked client and CLI tests
- mocked ingestion tests for webpage fetch and save behavior
- local module and smoke tests
- orchestration-level integration tests using temporary vaults and real local indexing/retrieval flow
- service-layer tests for UI-facing query and status responses
- phase-focused tests for retrieval, metadata, links, save-back behavior, and optional web search

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
- In `auto` mode, weak local retrieval may trigger external web fallback

### Web search is unavailable

Symptom:

- The app warns that web search was requested but unavailable.

Fix:

- Check your internet connection
- Verify `WEB_SEARCH_PROVIDER`
- Verify `WEB_SEARCH_API_URL` if you overrode the default
- Retry in `local_only` mode if you want to stay fully local

### Webpage ingestion fails

Symptom:

- The app says the webpage could not be fetched or parsed.

Fix:

- Check that the URL is valid and publicly reachable
- Retry the page in a browser to confirm it is available
- Some sites require JavaScript-heavy rendering or block simple fetch clients
- Increase `WEBPAGE_FETCH_TIMEOUT_SECONDS` if the site is slow to respond

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
- The UI exposes retrieval/debug structure intended to support future features, but it is still intentionally simple
- Web search is optional and external, so its availability and result quality depend on the configured provider
- The default provider is Wikipedia search, which is more reliable than the previous DuckDuckGo-only approach but is still narrower than a full general web search engine
- External web evidence is kept separate from local notes, but answer quality still depends on prompt quality and source quality
- Webpage ingestion uses lightweight HTML extraction, so some pages with heavy JavaScript rendering or aggressive boilerplate may not import cleanly
- Automated tests are strong locally, but live Ollama behavior is still mostly verified manually
- Prompting is intentionally simple

## Suggested V2 Roadmap

- Source snippet highlighting
- Configurable prompt templates
- Metadata filtering by tags, frontmatter, or note type
- Conversation history
- Richer UI features such as persistent sessions and better source inspection
- Optional live integration checks for Ollama and Chroma
- Additional ingestion sources such as YouTube transcripts through the same ingestion service layer
