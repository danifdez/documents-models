# RAG Pipeline

The Retrieval-Augmented Generation (RAG) system enables semantic search and AI-powered question answering over ingested documents. It involves three job types: `ingest-content`, `search`, and `ask`.

## Pipeline Overview

```
                         ┌──────────────┐
                         │  Document    │
                         │  Content     │
                         └──────┬───────┘
                                │
                         ingest-content
                                │
                   ┌────────────┼────────────┐
                   │            │            │
              Clean HTML    Chunk Text   Encode Chunks
              (text.py)  (semantic_chunk)  (e5 model)
                   │            │            │
                   └────────────┼────────────┘
                                │
                       Store in rag_chunks
                       (pgvector + metadata)
                                │
         ┌──────────────────────┼──────────────────────┐
         │                                             │
      search                                         ask
         │                                             │
   [Retriever]                                  [Retriever]
   Encode query                                 Encode question
   pgvector cosine search                       pgvector cosine search
         │                                             │
   [Reranker]                                   [Reranker]
   Deduplicate + sort                           Deduplicate + sort
         │                                             │
   Return ranked                            [ContextBuilder]
   results                                  Join chunks into text
                                                       │
                                             [PromptBuilder]
                                             Fill prompt template
                                                       │
                                              [Generator]
                                              LLM inference
                                                       │
                                              Return answer
```

## Content Ingestion (`ingest-content`)

The ingestion pipeline converts HTML content into searchable vector embeddings.

### Step 1: HTML Cleaning

The `clean_html_text()` function in `services/text.py` extracts text from block-level HTML elements:

- Removes `<script>` and `<style>` tags
- Extracts text from block elements: `p`, `div`, `h1`-`h6`, `li`, `td`, `th`, `blockquote`, `pre`, `section`, `article`, `header`, `footer`, `main`, `aside`
- Collapses whitespace within each element
- Captures any remaining text outside block elements

### Step 2: Semantic Chunking

The `semantic_chunk_text()` function uses recursive splitting with overlap:

1. Block elements are joined with paragraph separators (`\n\n`).
2. If the total text is under `RAG_CHUNK_MAX_WORDS`, it is returned as a single chunk.
3. Otherwise the text is recursively split by `\n\n`, `\n`, `. `, and finally by word count.
4. Segments are accumulated into chunks, carrying `RAG_CHUNK_OVERLAP_WORDS` words from the previous chunk.

Default values (configurable via env vars):

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_CHUNK_TARGET_WORDS` | `150` | Target words per chunk |
| `RAG_CHUNK_MAX_WORDS` | `250` | Split threshold |
| `RAG_CHUNK_OVERLAP_WORDS` | `30` | Overlap between consecutive chunks |

### Step 3: Embedding

Each chunk is encoded using the `EmbeddingService` singleton (`services/embedding_service.py`):

- **Model**: configured in `config/tasks.json` (default: `intfloat/multilingual-e5-small`, multilingual)
- **Dimensions**: 384
- **Normalization**: L2-normalized (`normalize_embeddings=True`)
- Batch encoding for efficiency

### Step 4: Storage

Old vectors for the source are deleted before upserting (automatic re-sync). Embeddings are stored as rows in the `rag_chunks` table (PostgreSQL/pgvector):

- **Row ID**: Random UUID
- **Vector**: 384-dimensional `vector` column
- **Columns**:
  - `text` — The original text chunk
  - `source_id` — Source identifier (resource ID, `doc_{id}`, or `knowledge_{id}`)
  - `source_type` — `resource`, `doc`, or `knowledge`
  - `project_id` — UUID of the project
  - `part_number` — Sequential chunk index (1-based)
  - `total_chunks` — Total number of chunks for this source

## Semantic Search (`search`)

The search pipeline is: **Retriever → Reranker**.

1. **Retriever** — Encodes the query using `encode_query()`, which applies a BGE instruction prefix (`"Represent this sentence: "`) for asymmetric retrieval. Queries `rag_chunks` (pgvector cosine) with optional `project_id` filter and `score_threshold`.
2. **Reranker** — Filters empty chunks and duplicates, then sorts by cosine score descending.
3. Returns up to `limit` results with `text`, `score`, and full `metadata`.

## Question Answering (`ask`)

The ask pipeline is: **Retriever → Reranker → ContextBuilder → PromptBuilder → Generator**.

1. **Retriever** — Same as search (encodes question with BGE prefix, retrieves top-k chunks).
2. **Reranker** — Deduplicates and sorts chunks by score.
3. **ContextBuilder** — Joins ranked chunks into a single context string (separator: `\n\n---\n\n`).
4. **PromptBuilder** — Fills the `ask` prompt template (from `templates/prompts/`) with context and question.
5. **Generator** — Runs LLM inference using the model configured for the `ask` task in `tasks.json`.

Default prompt instructs the LLM to:
- Answer only using the provided context
- Respond in the same language the question is asked
- Use at most `RAG_MAX_TOKENS` tokens

If no relevant chunks are found, returns `"No relevant information was found to answer this question."` without calling the LLM.

## Embedding Service

The `EmbeddingService` class (`services/embedding_service.py`) wraps sentence-transformers:

- Model loaded from `config/tasks.json` (task: `embedding`)
- Runs on GPU automatically when CUDA is available
- Provides:
  - `encode(texts)` — batch encoding for ingestion
  - `encode_single(text)` — single text encoding
  - `encode_query(text)` — query encoding with BGE instruction prefix (for asymmetric retrieval)
- All embeddings are L2-normalized by default
- Singleton instance shared across all tasks via `get_embedding_service()`

## LLM Service

The `LLMService` class (`services/llm_service.py`) wraps llama-cpp-python:

- Model path and parameters loaded from `config/tasks.json` via `get_llm_params(task_name)`
- Cached per model path via `get_llm_service()` — one instance shared across requests for the same model
- Provides `generate(prompt, max_tokens)` for completion and `chat(messages, max_tokens)` for chat completion

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DEFAULT_LIMIT` | `5` | Number of chunks retrieved for the `ask` task |
| `RAG_MAX_TOKENS` | `1000` | Maximum tokens in the LLM response |
| `RAG_SCORE_THRESHOLD` | `0.35` | Minimum similarity score for a chunk to be included |
| `RAG_CHUNK_TARGET_WORDS` | `150` | Target chunk size (words) |
| `RAG_CHUNK_MAX_WORDS` | `250` | Maximum words before splitting |
| `RAG_CHUNK_OVERLAP_WORDS` | `30` | Overlap words between consecutive chunks |

Vector table names are configured under the `vectors` block in `config/config.json` (default workspace table: `rag_chunks`). See [Configuration](./configuration.md) for the complete reference.
