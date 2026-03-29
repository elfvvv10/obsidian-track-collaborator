# Obsidian Track Collaborator

Obsidian Track Collaborator is a local-first music production assistant for electronic tracks. It turns an Obsidian vault into a working collaborator that can retrieve relevant notes, keep persistent memory for an in-progress track, and generate practical producer-facing output.

The project is built for real working sessions rather than generic chat. You can run it fully local with Ollama, add optional OpenAI-compatible chat generation, and use the same service layer through either the CLI or Streamlit UI.

## What It Helps With

- retrieving ideas, references, and notes from your vault
- keeping persistent memory for an active track with YAML Track Context
- carrying forward per-track tasks across sessions for continuity and prioritization
- critiquing sections, arrangements, and overall track direction
- generating practical output like bassline ideas, arrangement moves, and sound-design directions
- saving collaborator outputs back into your vault with clear source separation

## Quick Start

1. Install Python 3.11+ and [Ollama](https://ollama.com/).
2. Pull the default local models:

```bash
ollama pull deepseek-r1
ollama pull nomic-embed-text
```

3. Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Copy the sample environment:

```bash
cp .env.example .env
```

5. Build the local index:

```bash
python main.py index
```

6. Ask a question:

```bash
python main.py ask "How can I improve this bassline groove?"
```

7. Launch the UI when you want a longer working session:

```bash
streamlit run streamlit_app.py
```

Track-aware CLI example:

```bash
python main.py ask "How can I make this drop hit harder?" \
  --track-id moonlit_driver \
  --use-track-context \
  --section-focus drop
```

The included `sample_vault/` is set up for realistic local testing.

## Core Mental Model

- `Track Context`: persistent track memory for the active in-progress track. Use it for identity, current problem, known issues, goals, and section-aware context.
- `Track Tasks`: lightweight persisted per-track tasks. Use them to keep open work attached to the active track across sessions.
- `Arrangement notes`: structural timeline and section description over time. Use them for bars, energy, elements, and arrangement flow.
- `Saved Outputs`: generated collaborator artifacts such as answers, critiques, arrangement plans, and research notes.

The app keeps those layers separate so long-term memory, current work, structure, and generated outputs do not collapse into one note type.

## Main Workflows

- `General Ask`: grounded producer collaboration for normal questions
- `Track Concept Critique`: higher-signal critique for track direction and weak spots
- `Arrangement Planner`: section and timeline planning for a track
- `Sound Design Brainstorm`: practical patch, groove, and sound ideas
- `Research Session`: visible multi-step research across your vault and optional web context

All workflows keep local notes, imports, saved outputs, and web evidence labeled separately.

## CLI Examples

Basic ask:

```bash
python main.py ask "How can I improve this drop transition?"
```

Track-aware ask:

```bash
python main.py ask "How can I make the breakdown re-entry stronger?" \
  --track-id moonlit_driver \
  --use-track-context \
  --section-focus breakdown
```

Hybrid retrieval:

```bash
python main.py ask "Compare my notes with external context" --retrieval-mode hybrid
```

Strict evidence-only answer:

```bash
python main.py ask "Use only retrieved evidence" --answer-mode strict
```

Ingest a webpage:

```bash
python main.py ingest-webpage "https://example.com/article" --title "Example Article" --index-now
```

Ingest a YouTube video:

```bash
python main.py ingest-youtube "https://www.youtube.com/watch?v=example" --title "Example Video" --index-now
```

Ingest a PDF:

```bash
python main.py ingest-pdf "/path/to/notes.pdf" --title "PDF Notes" --index-now
```

Ingest a DOCX:

```bash
python main.py ingest-docx "/path/to/notes.docx" --title "DOCX Notes" --index-now
```

## Track Context

Track Context is central to the project. YAML Track Context is stored under:

```text
<OBSIDIAN_OUTPUT_PATH>/track_contexts/
```

It influences:

- weighted retrieval and reranking
- track-aware prompting
- query rewriting and final retrieval ranking
- section-aware critique specificity
- persisted per-track task loading for the active track
- reviewable Track Context update proposals
- current-track visibility in the UI

CLI support for the YAML flow:

- `--track-id` selects the active in-progress track
- `--use-track-context` loads YAML Track Context for the turn
- `--section-focus` carries a section such as `drop` or `breakdown` into the prompt
- when an answer includes a reviewable Track Context update, the CLI shows the proposal, previews the updated Track Context, and lets you choose whether to apply it to YAML

Per-track tasks are stored beside Track Context as separate YAML files:

```text
<OBSIDIAN_OUTPUT_PATH>/track_contexts/<track_id>.tasks.yaml
```

They are loaded automatically when the active YAML track is loaded. Only open tasks influence retrieval ranking, and they act as a small prioritization nudge rather than overriding stronger evidence.

Compatibility note: legacy markdown `Projects/Current Tracks/<Track Name>/track_context.md` is still tolerated, but YAML Track Context is the primary editable path.

## UI Overview

The Streamlit UI is built for longer working sessions. The most important pieces are:

- persistent YAML Track Context editing in the sidebar
- current-track visibility close to the composer
- workflow-aware asking and critique flows
- manual review and apply for suggested Track Context updates
- debug panels for retrieval, rewritten queries, and evidence use
- model/provider selection for the active session

## Providers

Default behavior is fully local:

- `CHAT_PROVIDER=ollama`
- `EMBEDDING_PROVIDER=ollama`
- `OLLAMA_CHAT_MODEL=deepseek`
- `OLLAMA_EMBEDDING_MODEL=nomic-embed-text`

Optional hybrid mode uses OpenAI-compatible chat generation while keeping embeddings local:

- `CHAT_PROVIDER=openai`
- `EMBEDDING_PROVIDER=ollama`
- `OPENAI_CHAT_MODEL=gpt-4.1-mini`

For the full configuration surface, see [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## Save-Back

Generated collaborator outputs are saved under `Saved Outputs/`, including:

- `answers/General Asks/`
- `answers/Arrangement Plans/`
- `answers/Sound Design Brainstorms/`
- `critiques/Genre Fit Reviews/`
- `critiques/Track Concept Critiques/`
- `research/`

Saved notes can include:

- workflow and domain metadata
- Track Context summaries
- active section focus and track-linked context metadata
- reviewable Track Context update metadata
- clearly labeled sources

Imported source notes can also be saved under configured vault folders such as:

- `Imports/Web Imports/`
- `Imports/YouTube Imports/`
- `Imports/PDF Imports/`
- `Imports/Word Imports/`

## Project Boundaries

What the project already does well:

- local retrieval over a structured Obsidian vault
- weighted reranking shaped by track context, section focus, and open track tasks
- workflow-aware producer collaboration
- persistent Track Context memory
- persisted per-track task continuity
- arrangement-aware critique support
- structured webpage and video ingestion
- local-first operation with optional OpenAI-compatible chat

What is intentionally not implemented yet:

- OpenAI embeddings
- automatic Track Context mutation without review
- automatic assistant task persistence without review
- autonomous critique scoring engines
- a major UI redesign

## Further Reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): service layout, retrieval pipeline, and subsystem overview
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md): environment variables, providers, indexing, retrieval, and ingestion settings
- [`VAULT_ORGANIZATION.md`](VAULT_ORGANIZATION.md): deeper vault layout guidance
- [`sample_vault/Saved Outputs/README.md`](sample_vault/Saved%20Outputs/README.md): saved-output conventions
- [`sample_vault/Templates/README.md`](sample_vault/Templates/README.md): reusable sample templates

## Tests

Run the full suite:

```bash
python3 -m unittest discover -s tests
```

You can also double-click `run.command` on macOS to activate `.venv` and start Streamlit quickly.
