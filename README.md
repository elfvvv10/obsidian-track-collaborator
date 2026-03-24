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
- Supports answer modes for stricter or more exploratory answer behavior
- Supports a visible research mode that decomposes a goal into explicit subquestions
- Labels local sources, web sources, and inference more explicitly
- Shows source note references in the terminal
- Includes a lightweight local Streamlit UI for asking questions, indexing, and debugging
- Supports optional external web search as a separate evidence path
- Supports webpage ingestion as a separate content-import workflow
- Supports YouTube transcript ingestion as a separate content-import workflow
- Optionally saves answers back into the vault as Markdown notes
- Uses incremental indexing to update only changed notes
- Excludes saved answers in the configured output folder from indexing by default when that folder lives inside the vault
- Can optionally index saved answers as secondary retrieval sources with distinct `[Saved N]` labels
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

`YouTube URL -> ingestion service -> transcript fetch -> Markdown note in vault -> optional indexing`

The app now also includes a thin service layer so both the CLI and UI can share the same orchestration path without duplicating business logic.

Research mode is handled as a separate orchestration path above normal direct Q&A:

`Goal -> research service -> subquestion plan -> repeated query/answer steps -> final synthesis -> optional save-back`

Core modules:

- `main.py`: CLI entrypoint
- `config.py`: environment loading and validation
- `services/index_service.py`: shared indexing/build flow for CLI and UI
- `services/query_service.py`: shared query + answer flow for CLI and UI
- `services/research_service.py`: visible multi-step research workflow built on top of the normal query stack
- `services/web_search_service.py`: optional external search orchestration
- `services/web_alignment_service.py`: local-guided web query building and off-topic result filtering
- `services/ingestion_service.py`: shared external content ingestion orchestration
- `services/models.py`: structured request/response models for service consumers
- `services/common.py`: shared service helpers such as link resolution and index checks
- `services/ingestion_helpers.py`: shared note-building and collision-safe save helpers for imported content
- `services/webpage_ingestion_service.py`: webpage fetch, text extraction, and note creation
- `services/youtube_ingestion_service.py`: YouTube transcript retrieval and note creation
- `services/prompt_service.py`: answer-mode prompt policies, citation labels, and inference guidance
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
INDEX_SAVED_ANSWERS=false
WEB_SEARCH_PROVIDER=wikipedia
WEB_SEARCH_API_URL=
WEB_SEARCH_MAX_RESULTS=3
WEB_SEARCH_TIMEOUT_SECONDS=10
WEBPAGE_INGESTION_FOLDER=ingested_webpages
YOUTUBE_INGESTION_FOLDER=ingested_youtube
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
- `INDEX_SAVED_ANSWERS`: when enabled, saved answers in the output folder are indexed as secondary derived notes
- `WEB_SEARCH_PROVIDER`: external search provider. `wikipedia` is the default no-key option. `duckduckgo` remains available as an alternative.
- `WEB_SEARCH_API_URL`: optional provider endpoint override. Leave blank to use the provider default.
- `WEB_SEARCH_MAX_RESULTS`: maximum number of external results to include
- `WEB_SEARCH_TIMEOUT_SECONDS`: timeout for external web search requests
- `WEBPAGE_INGESTION_FOLDER`: vault-relative folder where ingested webpages are saved
- `YOUTUBE_INGESTION_FOLDER`: vault-relative folder where ingested YouTube transcript notes are saved
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
python main.py ask "What do my notes say about AI agents?" --answer-mode strict
python main.py ask "Compare my notes with recent external context" --retrieval-mode hybrid --answer-mode exploratory
```

The app will:

1. Embed your question locally
2. Retrieve the top matching note chunks from ChromaDB
3. Send the retrieved context and your question to Ollama
4. Print the answer and sources in the terminal
5. Ask whether you want to save the answer as a Markdown note

## Research Mode

Research mode keeps the current direct ask flow intact, but adds a more structured workflow for bigger questions:

1. interpret the goal
2. generate a small set of explicit subquestions
3. answer each subquestion through the existing retrieval and answer stack
4. synthesize a final research answer
5. keep warnings, sources, and inference labels visible

Run it from the CLI:

```bash
python main.py research "Compare my notes on AI agents with recent external context"
python main.py research "What do my notes suggest about local LLM workflows?" --answer-mode strict --max-subquestions 2
```

Research mode differs from direct ask mode in two important ways:

- it is multi-step and inspectable rather than a single retrieval-and-answer pass
- it reuses the existing query stack for each subquestion instead of hiding everything inside one large model call

It is still intentionally bounded:

- only a small number of subquestions are generated
- there is no uncontrolled autonomous looping
- answer mode and retrieval mode still apply

### Retrieval Modes

- `local_only`: use only your Obsidian vault. This is the default and preserves the original local-first behavior.
- `auto`: try local retrieval first, and use web search only when local evidence is missing or weak.
- `hybrid`: use local retrieval and web search together, even if local evidence exists.

When web search is used, local note sources and web sources are labeled separately in both the CLI and UI.
The default provider is now Wikipedia search because it is a more reliable no-key option than the previous DuckDuckGo-only path.
When local note evidence exists in `hybrid` mode, the app now narrows the web query using the strongest local note topics and filters out clearly off-topic web results before prompting.
For the default Wikipedia provider, broad hybrid prompts are rewritten into more concrete topic-oriented web queries so the external lookup behaves more like a topical reference search.
If that first local-guided web query returns no provider results, the app makes one lighter retry using the strongest local title or heading anchor before giving up on web evidence.
Warnings now distinguish between provider failures, zero provider results, and results that were discarded as off-topic.

### Answer Modes

- `strict`: use only retrieved evidence, prefer refusal when support is missing, and enforce the strongest citation discipline.
- `balanced`: evidence first, with limited reasoning to connect supported ideas. If the answer goes beyond direct evidence, it should be labeled with `[Inference]`.
- `exploratory`: evidence plus broader synthesis and extrapolation, with inference explicitly labeled using `[Inference]`.

Answer mode controls how the model is allowed to write the answer. Retrieval mode controls which evidence sources are available in the first place.

### Source Labels and Inference

- `[Local 1]`, `[Local 2]`, and so on refer to your Obsidian notes.
- `[Web 1]`, `[Web 2]`, and so on refer to external web results.
- `[Inference]` marks model synthesis that goes beyond directly retrieved evidence.

If the model does not include citation labels in the generated text, the app adds a short evidence summary so the used sources are still explicit.

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

## Ingest YouTube Videos

YouTube ingestion is also an import workflow. It differs from webpage ingestion because it focuses on transcript text instead of HTML page extraction, and it differs from query-time web search because it creates a permanent vault note rather than temporary answer-time evidence.

Run it from the CLI:

```bash
python main.py ingest-youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python main.py ingest-youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --title "Example Video" --index-now
```

Saved YouTube notes go into `YOUTUBE_INGESTION_FOLDER` inside your vault, which defaults to `ingested_youtube/`.

Each saved note includes:

- video title when available
- source URL
- YouTube video ID
- ingestion timestamp
- transcript text

## Run the Local UI

The Streamlit UI uses the same service layer as the CLI, so indexing and question-answering behavior stay aligned.

Start the UI with:

```bash
streamlit run streamlit_app.py
```

The UI includes four main areas:

- `Sidebar`: query filters and retrieval controls such as folder, path text, tag, top-k, reranking, linked-note expansion, auto-save, retrieval mode, and answer mode
- `Ask`: question input, a visible workflow toggle for `Direct Ask` or `Research Mode`, answer display, separate local/web sources, save actions, linked-note context, and an optional debug view of retrieval stages
- `Ask`: when web search is attempted, the UI can also show the actual web query used, whether a retry was attempted, and a brief explanation when no web sources were included
- `Ask`: when saved answers are indexed and used, they appear in a separate `Saved Answer Sources` section
- `Ask`: a visible toggle directly under the question box lets you decide per question whether indexed saved answers should be included
- `Ask`: in research mode, the UI shows the generated subquestions, step-by-step findings, and the final synthesized answer
- `Ingest`: paste a webpage URL or YouTube URL, save it into the vault, and optionally trigger indexing right away
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

Saved answer notes also include lightweight frontmatter metadata such as `source_type: "saved_answer"`, the original question, and the save timestamp so they can be recognized safely later if you enable indexing for them.

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

If you enable `INDEX_SAVED_ANSWERS=true`, those saved notes are indexed as secondary derived sources. They are labeled separately as `[Saved N]` and are down-ranked relative to primary vault notes so they can help recall without overtaking the original notes they summarize.

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
│   ├── ingestion_helpers.py
│   ├── index_service.py
│   ├── models.py
│   ├── query_service.py
│   ├── research_service.py
│   ├── web_search_service.py
│   ├── webpage_ingestion_service.py
│   └── youtube_ingestion_service.py
├── requirements.txt
├── .env.example
├── README.md
├── sample_vault/
└── tests/
```

The `tests/` directory includes:

- mocked client and CLI tests
- mocked ingestion tests for webpage fetch and save behavior
- mocked ingestion tests for webpage and YouTube import behavior
- local module and smoke tests
- orchestration-level integration tests using temporary vaults and real local indexing/retrieval flow
- service-layer tests for UI-facing query and status responses
- visible research-workflow tests for subquestion planning, step execution, and final synthesis
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

If web search was attempted but no web sources were used, check the warning details:

- the provider may have returned no results for the guided query
- returned results may have been discarded because they did not align with the strongest local note topics
- a lighter retry may already have been attempted automatically

### Webpage ingestion fails

Symptom:

- The app says the webpage could not be fetched or parsed.

Fix:

- Check that the URL is valid and publicly reachable
- Retry the page in a browser to confirm it is available
- Some sites require JavaScript-heavy rendering or block simple fetch clients
- Increase `WEBPAGE_FETCH_TIMEOUT_SECONDS` if the site is slow to respond

### YouTube ingestion fails

Symptom:

- The app says the YouTube transcript could not be retrieved.

Fix:

- Check that the URL is a valid YouTube watch, short, or youtu.be link
- Confirm the video has transcripts available
- Install dependencies from `requirements.txt` so `youtube-transcript-api` is present
- Try a different video if transcripts are disabled or unavailable

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
- Saved answers can be indexed as a secondary source and toggled per question in the UI, but they are still a lightweight derived-note feature rather than a full memo-management workflow
- The Streamlit UI is intentionally lightweight and does not yet include persistent chat history or advanced source inspection workflows
- The UI exposes retrieval/debug structure intended to support future features, but it is still intentionally simple
- Research mode is intentionally bounded to a small number of explicit subquestions and does not yet support deeper branching, follow-up planning, or iterative revision
- Web search is optional and external, so its availability and result quality depend on the configured provider
- The default provider is Wikipedia search, which is more reliable than the previous DuckDuckGo-only approach but is still narrower than a full general web search engine
- External web evidence is kept separate from local notes, but answer quality still depends on prompt quality and source quality
- In `hybrid` and weak-`auto` cases, web search is intentionally conservative and may return no web evidence if aligned results cannot be found
- Hallucination guards are lightweight and rule-based; they reduce unsupported answers but are not a full verification system
- Citation enforcement is prompt- and formatting-based rather than a strict post-hoc fact checker
- Webpage ingestion uses lightweight HTML extraction, so some pages with heavy JavaScript rendering or aggressive boilerplate may not import cleanly
- YouTube ingestion depends on transcript availability and will not work well for videos without transcripts
- Automated tests are strong locally, but live Ollama behavior is still mostly verified manually
- Prompting is intentionally simple

## Suggested V2 Roadmap

- Source snippet highlighting
- Configurable prompt templates
- Metadata filtering by tags, frontmatter, or note type
- Conversation history
- Richer research workflows such as follow-up planning, evidence comparison, and revision passes
- Richer UI features such as persistent sessions and better source inspection
- Optional live integration checks for Ollama and Chroma
- Better cleanup, sectioning, and summarization options for imported external content
