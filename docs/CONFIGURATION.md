# Configuration Reference

## Default Local Setup

The default setup keeps both chat and embeddings local through Ollama:

```env
CHAT_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_CHAT_MODEL=deepseek
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Hybrid Chat Setup

Optional hybrid mode uses OpenAI-compatible chat generation with local embeddings:

```env
CHAT_PROVIDER=openai
EMBEDDING_PROVIDER=ollama
OPENAI_API_KEY=your_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

If `CHAT_PROVIDER=openai` is selected, both `OPENAI_API_KEY` and `OPENAI_CHAT_MODEL` must be set.

## Example Environment

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
PDF_INGESTION_FOLDER=Imports/PDF Imports
DOCX_INGESTION_FOLDER=Imports/Word Imports
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
WEBPAGE_FETCH_USER_AGENT=obsidian-track-collaborator/1.0

TRACK_CRITIQUE_FRAMEWORK_PATH=
FRAMEWORK_DEBUG=false
```

## Important Paths

- `OBSIDIAN_VAULT_PATH`: source vault for notes and knowledge
- `OBSIDIAN_OUTPUT_PATH`: generated outputs and YAML Track Context storage
- `CHROMA_DB_PATH`: local vector store path

YAML Track Context lives under:

```text
<OBSIDIAN_OUTPUT_PATH>/track_contexts/
```

## Retrieval Controls

- `TOP_K_RESULTS`: number of final chunks used when answering
- `RETRIEVAL_CANDIDATE_MULTIPLIER`: candidate pool expansion before final selection
- `ENABLE_RERANKING`: enable heuristic reranking
- `ENABLE_LINKED_NOTE_EXPANSION`: include linked note context

At runtime, the CLI also supports:

- `--retrieval-scope`
- `--retrieval-mode`
- `--answer-mode`
- `--boost-tag`
- `--include-linked`

## Import and Index Controls

- `INDEX_WEBPAGE_IMPORTS`: include webpage imports in indexing
- `INDEX_YOUTUBE_IMPORTS`: include video imports in indexing
- `INDEX_PDF_IMPORTS`: include PDF imports in indexing
- `INDEX_DOCX_IMPORTS`: include DOCX imports in indexing
- `AUTO_INDEX_AFTER_INGESTION`: trigger incremental indexing after import
- `YOUTUBE_INDEX_MODE=sections`: index video imports by semantic sections

## Save-Back Controls

- `AUTO_SAVE_ANSWER`: save answers automatically
- `INDEX_SAVED_ANSWERS`: allow generated outputs to become retrievable local material
- `RESEARCH_SESSIONS_FOLDER`: destination for saved research outputs

Saved collaborator outputs are written under `Saved Outputs/` and may include:

- workflow metadata
- Track Context summaries
- reviewable Track Context update metadata
- labeled sources
