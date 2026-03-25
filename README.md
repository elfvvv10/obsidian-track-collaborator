# Obsidian Electronic Music Research Assistant

A local-first Obsidian RAG assistant for electronic music research, track development, critique, arrangement analysis, sound design ideation, and producer-focused knowledge retrieval.

It combines:
- a local Obsidian vault
- ChromaDB for vector search
- Ollama for local embeddings and optional local chat
- an optional OpenAI-compatible chat provider for answer generation
- a Streamlit UI plus a CLI that share the same service-layer orchestration

## Quick Start

### Local-first default

1. Install Python 3.11+ and [Ollama](https://ollama.com/).
2. Pull the default local models:

```bash
ollama pull deepseek-r1
ollama pull nomic-embed-text
```

3. Clone the repo and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Copy the sample environment:

```bash
cp .env.example .env
```

5. Build the index:

```bash
python main.py index
```

6. Ask a question:

```bash
python main.py ask "How can I improve this bassline groove?"
```

7. Or launch the UI:

```bash
streamlit run streamlit_app.py
```

You can also double-click [run.command](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/run.command) on macOS to activate `.venv` and start Streamlit quickly.

The included `sample_vault/` is set up for realistic local testing.

## What The App Does

- loads Markdown notes from an Obsidian vault
- chunks notes into retrieval-friendly units
- stores embeddings in local ChromaDB
- retrieves relevant evidence for questions
- keeps local notes, imported material, saved outputs, and web evidence clearly separated
- supports optional web fallback and hybrid retrieval
- supports producer-facing workflows such as critique, arrangement planning, and sound design brainstorming
- supports persistent YAML Track Context for long-term track memory
- supports arrangement notes as structured `track_arrangement` knowledge objects
- supports webpage and YouTube/video knowledge ingestion into the vault
- supports answer save-back into `Saved Outputs/`
- supports both CLI and Streamlit UI through the same service layer

## Current Architecture

The main path is:

`vault -> loader -> chunker -> embeddings -> ChromaDB -> retriever -> prompt service -> chat model -> answer/save-back`

The app is no longer a single-script RAG prototype. It now has a shared service layer with explicit orchestration for:

- indexing
- direct question answering
- research sessions
- workflow-aware prompting
- ingestion
- Track Context persistence
- retrieval-only query rewriting
- save-back

Core modules:

- [main.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/main.py): CLI entrypoint
- [streamlit_app.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/streamlit_app.py): local UI
- [config.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/config.py): environment loading and validation
- [model_clients.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/model_clients.py): provider-agnostic chat/embedding interfaces
- [model_provider.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/model_provider.py): provider-aware client construction
- [llm.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/llm.py): Ollama and OpenAI-compatible chat clients
- [embeddings.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/embeddings.py): Ollama embedding client
- [services/query_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/query_service.py): direct ask orchestration
- [services/research_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/research_service.py): visible multi-step research flow
- [services/prompt_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/prompt_service.py): answer policies, evidence formatting, workflow-aware prompts
- [services/index_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/index_service.py): indexing/build flow
- [services/models.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/models.py): structured request/response/domain models
- [services/music_workflow_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/music_workflow_service.py): workflow routing and save destinations
- [services/track_context_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/track_context_service.py): YAML Track Context persistence plus legacy markdown loading
- [services/arrangement_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/arrangement_service.py): arrangement note parsing/rendering
- [services/video_ingestion_service.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/services/video_ingestion_service.py): structured video knowledge import pipeline
- [chunker.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/chunker.py): general plus arrangement/video-aware chunking
- [retriever.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/retriever.py): retrieval and linked-note expansion
- [vector_store.py](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/vector_store.py): ChromaDB persistence

## Provider Setup

The app now supports a provider split:

- chat generation: `ollama` or `openai`
- embeddings: currently `ollama`

Default behavior remains fully local:

```env
CHAT_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_CHAT_MODEL=deepseek
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Hybrid mode

The first API-backed hybrid setup is:

```env
CHAT_PROVIDER=openai
EMBEDDING_PROVIDER=ollama
OPENAI_API_KEY=your_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

That gives you:
- OpenAI-compatible chat generation
- local Ollama embeddings
- local ChromaDB retrieval
- unchanged retrieval/index/save-back architecture

If `CHAT_PROVIDER=openai` but `OPENAI_API_KEY` or `OPENAI_CHAT_MODEL` is missing, the app fails clearly at chat-client construction time.

## Environment

Example `.env`:

```env
OBSIDIAN_VAULT_PATH=./sample_vault
OBSIDIAN_OUTPUT_PATH=./sample_vault/Saved Outputs
CHROMA_DB_PATH=./chroma_db

CHAT_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama

OPENAI_API_KEY=
OPENAI_CHAT_MODEL=
OPENAI_BASE_URL=https://api.openai.com/v1

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek
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
RESEARCH_SESSIONS_FOLDER=Saved Outputs/research
CURATED_KNOWLEDGE_FOLDER=Knowledge
INDEX_RESEARCH_SESSIONS=false
INDEX_WEBPAGE_IMPORTS=true
INDEX_YOUTUBE_IMPORTS=true

WEB_SEARCH_PROVIDER=wikipedia
WEB_SEARCH_API_URL=
WEB_SEARCH_MAX_RESULTS=3
WEB_SEARCH_TIMEOUT_SECONDS=10

WEBPAGE_INGESTION_FOLDER=Imports/Web Imports
YOUTUBE_INGESTION_FOLDER=Imports/YouTube Imports
AUTO_INDEX_AFTER_INGESTION=true

YOUTUBE_WHISPER_MODEL=small
YOUTUBE_MAX_DURATION_SECONDS=7200
YOUTUBE_SEMANTIC_CHUNK_TARGET_SECONDS=120
YOUTUBE_SEMANTIC_CHUNK_TARGET_CHARS=1200
YOUTUBE_TEMP_DIR=
YOUTUBE_SAVE_MARKDOWN_IMPORT_NOTE=true
YOUTUBE_INDEX_MODE=sections
YOUTUBE_ALLOW_TRANSCRIPT_FALLBACK=true

WEBPAGE_FETCH_TIMEOUT_SECONDS=15
WEBPAGE_FETCH_USER_AGENT=obsidian-rag-assistant/1.0

TRACK_CRITIQUE_FRAMEWORK_PATH=
FRAMEWORK_DEBUG=false
```

Key notes:

- `OBSIDIAN_OUTPUT_PATH` now points to `Saved Outputs`
- webpage and YouTube imports are indexed by default in the sample config
- the sample config and local `.env.example` are intentionally aligned

## Indexing

Build the local index:

```bash
python main.py index
```

Rebuild from scratch:

```bash
python main.py rebuild
```

Optional overrides:

```bash
python main.py index --chunk-size 800 --chunk-overlap 100
python main.py index --chunking-strategy sentence
```

Indexing is incremental:
- new notes are added
- changed notes are re-embedded
- deleted notes are removed

## Asking Questions

CLI ask:

```bash
python main.py ask "How can I improve this drop transition?"
```

Examples:

```bash
python main.py ask "How can I improve this drop transition?" --retrieval-scope knowledge
python main.py ask "Search working notes too" --retrieval-scope extended
python main.py ask "Compare my notes with external context" --retrieval-mode hybrid
python main.py ask "Use only retrieved evidence" --answer-mode strict
python main.py ask "Be more exploratory" --answer-mode exploratory
python main.py ask "Boost notes tagged progressive-house" --boost-tag progressive-house
```

Retrieval modes:

- `local_only`: local vault evidence only
- `auto`: local first, web fallback when local evidence is weak/missing
- `hybrid`: local plus web

Retrieval scopes:

- `knowledge`: curated knowledge plus indexed imports
- `extended`: knowledge plus working notes and saved outputs

Answer modes:

- `strict`: evidence-bound
- `balanced`: evidence first with limited synthesis
- `exploratory`: broader synthesis with `[Inference]`

## Music Collaboration Workflows

The app is specialized for electronic music production and collaboration.

Current workflows:

- `General Ask`
- `Genre Fit Review`
- `Track Concept Critique`
- `Arrangement Planner`
- `Sound Design Brainstorm`
- `Research Session`

These workflows share the same trust boundaries:

- retrieval scope controls eligible local material
- retrieval mode controls web usage
- answer mode controls how far the model can reason beyond direct evidence
- local, imported, saved, and web sources stay labeled separately

## Track Context

Track Context is now intentionally narrow and persistent. It is long-term track memory, not a junk drawer for session notes or arrangement structure.

The primary YAML Track Context fields are:

- `track_id`
- `track_name`
- `genre`
- `bpm`
- `key`
- `vibe`
- `reference_tracks`
- `current_stage`
- `current_problem`
- `known_issues`
- `goals`

YAML Track Context is stored under:

```text
<OBSIDIAN_OUTPUT_PATH>/track_contexts/
```

Track Context currently influences:

- track-aware prompt context
- retrieval-only query rewriting
- critique-mode specificity
- assistant-suggested Track Context updates
- current-track visibility in the Streamlit UI

Legacy markdown `track_context.md` is still supported through the older workflow path flow:

```text
Projects/Current Tracks/<Track Name>/track_context.md
```

Rules:

- YAML Track Context is the primary editable path
- legacy markdown Track Context remains backward-compatible
- they stay separate
- only one Track Context source is injected into a prompt at a time

## Arrangement Notes

Arrangement is now a separate structured concept from Track Context.

Use arrangement notes for structural track description over time, not for persistent identity or session scratchpad text.

Arrangement notes use:

```yaml
type: track_arrangement
```

Recommended locations:

- active track work:
  - `Projects/Current Tracks/<Track Name>/arrangement.md`
  - or `Projects/Current Tracks/<Track Name>/arrangements/arrangement.md`
- reusable arrangement studies/reference analyses:
  - `Knowledge/Arrangement/<Example Name>.md`

The arrangement model supports:

- track metadata
- total bars
- ordered sections
- section names
- start/end bars
- energy
- elements
- notes
- issues
- purpose

There is a starter template in:

- [sample_vault/Templates/track_arrangement_template.md](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/sample_vault/Templates/track_arrangement_template.md)

## Critique Mode

`Track Concept Critique` now has a stronger prompt path that uses:

- Track Context for long-term track identity and current state
- arrangement evidence for section-aware structural analysis when available
- retrieved knowledge/import/reference material as evidence

Critique mode now guides the model toward a consistent response shape:

- Overall Assessment
- Arrangement / Energy Flow
- Genre / Style Fit
- Groove / Bass / Element Evolution
- Priority Issues
- Recommended Next Changes

If arrangement evidence is available, critique is asked to be section-aware and refer to section names/bars where practical.

If arrangement evidence is missing, critique falls back gracefully to a higher-level assessment instead of failing.

## Imports

### Webpage imports

CLI:

```bash
python main.py ingest-webpage "https://example.com/article"
python main.py ingest-webpage "https://example.com/article" --title "Example Article" --index-now
```

### YouTube / video imports

CLI:

```bash
python main.py ingest-youtube "https://www.youtube.com/watch?v=example"
python main.py ingest-youtube "https://www.youtube.com/watch?v=example" --title "Example Video" --index-now
```

The current video import path is no longer transcript-only. It now:

- extracts or retrieves transcript/audio information
- builds timestamped semantic sections
- saves a structured `youtube_video` markdown note
- preserves retrieval-friendly metadata
- reindexes from the saved markdown representation

Genre-aware import organization is supported in the UI:

- `Imports/Web Imports/<Genre>/...`
- `Imports/YouTube Imports/<Genre>/...`

`Generic` is always available for non-genre-specific imports.

When YAML Track Context has a genre, imported retrieval includes:

- `Generic`
- the matching import-genre bucket

## Streamlit UI

The Streamlit app is meant for real working sessions, not just quick demos.

Main areas:

- `Ask`
- `Ingest`
- `Index`
- `Settings / Debug`

UI highlights:

- persistent YAML Track Context sidebar editor
- legacy markdown workflow context selector/path flow
- clear current-track summary
- critique-context visibility for track-aware vs arrangement-aware critique
- session-level chat model selection
- suggested Track Context updates with manual apply
- debug separation between original question and rewritten retrieval query
- imported-genre retrieval visibility in debug mode

The UI keeps:

- normal mode relatively uncluttered
- debug details behind expandable panels
- current track and workflow context close to the composer

## Save-Back

Saved answers now go to:

- `Saved Outputs/answers/General Asks/`
- `Saved Outputs/critiques/Genre Fit Reviews/`
- `Saved Outputs/critiques/Track Concept Critiques/`
- `Saved Outputs/answers/Arrangement Plans/`
- `Saved Outputs/answers/Sound Design Brainstorms/`
- `Saved Outputs/research/`

Saved outputs include:

- question
- workflow/domain metadata
- structured workflow input when present
- Track Context summary when present
- answer
- sources
- light frontmatter for later indexing/classification

## Recommended Vault Layout

Recommended structure:

```text
Projects/
  Current Tracks/
    <Track Name>/
      track_context.md
      session_notes/
      arrangements/
      sound_design/
      exports/
  Track Ideas/
  Archived Tracks/

Knowledge/
  Arrangement/
  Drums and Groove/
  Genres/
  Mixing/
  References/
  Sound Design/

Imports/
  Web Imports/
  YouTube Imports/

Saved Outputs/
  answers/
    General Asks/
    Arrangement Plans/
    Sound Design Brainstorms/
  critiques/
    Genre Fit Reviews/
    Track Concept Critiques/
  research/

Templates/
  track_context_template.md
  track_arrangement_template.md
  video_import_template.md
  session_note_template.md

Sources/
  Frameworks/
    Music Production/
      track_critique_framework_v1.md

Archive/
```

For a fuller explanation, see [VAULT_ORGANIZATION.md](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/VAULT_ORGANIZATION.md).

## Templates

The sample vault includes reusable starters in [sample_vault/Templates](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/sample_vault/Templates):

- `track_context_template.md`
- `track_arrangement_template.md`
- `video_import_template.md`
- `session_note_template.md`

Framework documents such as the critique framework now live under [sample_vault/Sources/Frameworks](/Users/andrewhaynes/Coding/Obsidian%20RAG%20Assistant/sample_vault/Sources/Frameworks), for example:

- `Music Production/track_critique_framework_v1.md`

## Tests

Run the full suite:

```bash
python3 -m unittest discover -s tests
```

The test suite covers:

- client/provider behavior
- indexing and retrieval
- prompt construction
- Track Context
- arrangement parsing/chunking
- import genre routing
- webpage and video ingestion
- workflow orchestration
- UI/session helpers

## Current Boundaries

What the app already does well:

- local retrieval over a structured vault
- workflow-aware producer collaboration
- track-aware prompting
- arrangement-aware critique support
- structured video knowledge imports
- hybrid chat-provider support with local embeddings

What is intentionally not done yet:

- OpenAI embeddings
- a full provider-neutral readiness/status framework
- automatic Track Context mutation without review
- autonomous critique scoring/rules engines
- major UI redesign

## Notes

- Ollama remains the default and safest setup.
- OpenAI chat is now supported for generation, but embeddings remain local by design.
- The app is biased toward additive, conservative evolution: old markdown track context still works, old vault layouts can still be tolerated, and newer structured systems sit on top of the same core retrieval pipeline.
