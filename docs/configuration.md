# Configuration

All configuration is managed through environment variables with sensible defaults. Values are read in `config.py` using `os.getenv()`. Copy `.env.example` to `.env` for local development.

Per-task model selection is controlled separately via `config/models.json` — see [Per-task model configuration](#per-task-model-configuration) below.

## Environment Variables

### PostgreSQL

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `database` | PostgreSQL host (Docker service name by default) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `documents` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `example` | Database password |
| `JOBS_TABLE` | `jobs` | Name of the jobs table |

### Qdrant

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `qdrant` | Qdrant host (Docker service name by default) |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_URL` | `http://{QDRANT_HOST}:{QDRANT_PORT}` | Full Qdrant URL (auto-constructed from host and port if not set) |
| `QDRANT_COLLECTION` | `rag_docs` | Vector collection name |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DEFAULT_LIMIT` | `5` | Number of chunks retrieved for the `ask` task |
| `RAG_MAX_TOKENS` | `1000` | Maximum tokens in LLM responses for RAG |
| `RAG_SCORE_THRESHOLD` | `0.35` | Minimum cosine similarity score to include a chunk in results |

### RAG Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_CHUNK_TARGET_WORDS` | `150` | Target chunk size in words |
| `RAG_CHUNK_MAX_WORDS` | `250` | Maximum words before a chunk is split |
| `RAG_CHUNK_OVERLAP_WORDS` | `30` | Number of words shared between consecutive chunks |

### Worker Identity

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_ID` | auto UUID (persisted to `.worker_id`) | Stable worker identity across restarts. Set explicitly for predictable IDs |
| `WORKER_NAME` | `worker-{id[:8]}` | Human-readable name shown in logs and the `workers` table |
| `HEARTBEAT_INTERVAL` | `15` | Seconds between heartbeat updates |

### Worker Capabilities

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_DISABLE_LLM` | `false` | Set `true` to skip all LLM tasks (`keywords`, `key-point`, `ask`) |
| `WORKER_DISABLE_EMBEDDINGS` | `false` | Set `true` to skip all embedding tasks (`embedding`, `search`, `ingest-content`, `ask`) |

### Task Filtering

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_ENABLED_TASKS` | _(empty — all)_ | Comma-separated allowlist of task types this worker will process |
| `WORKER_DISABLED_TASKS` | _(empty — none)_ | Comma-separated blocklist of task types this worker will skip |

Both filters are applied after capability checks. `WORKER_ENABLED_TASKS` takes priority: if set, only listed tasks are eligible.

### Background Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKGROUND_HOURS_START` | `2` | Hour (0-23, inclusive) when the background window opens |
| `BACKGROUND_HOURS_END` | `6` | Hour (0-23, exclusive) when the background window closes |

`background` priority jobs are processed only when no `high`/`normal` jobs are pending, **or** when the current time falls inside the background window.

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENTS_STORAGE_DIR` | `../documents` | Root directory for uploaded documents |

## Per-task Model Configuration

Model selection for each task is driven by `config/models.json`. This file is auto-created on first run from `templates/models.default.json`. Edit it to change which model a task uses without modifying the code.

Default configuration (`templates/models.default.json`):

```json
{
  "llm_defaults": {
    "model_dir": "models",
    "n_ctx": 32768,
    "n_threads": 4,
    "n_batch": 64,
    "n_gpu_layers": 0
  },
  "tasks": {
    "keywords":          { "type": "llm",               "model": "Phi-4-mini-instruct-Q4_K_M.gguf" },
    "key-point":         { "type": "llm",               "model": "Phi-4-mini-instruct-Q4_K_M.gguf" },
    "ask":               { "type": "llm",               "model": "Phi-4-mini-instruct-Q4_K_M.gguf" },
    "embedding":         { "type": "sentence-transformer", "model": "BAAI/bge-small-en-v1.5" },
    "summarize":         { "type": "seq2seq",           "model": "facebook/mbart-large-50-one-to-many-mmt" },
    "translate":         { "type": "translation",       "model_prefix": "Helsinki-NLP/opus-mt" },
    "entity-extraction": { "type": "spacy",             "model": "en_core_web_sm" }
  }
}
```

LLM parameters (`n_ctx`, `n_threads`, `n_batch`, `n_gpu_layers`) can be overridden per-task by adding entries inside the task object.

> **Note:** `n_gpu_layers` is set to `0` in defaults. Override to `-1` to offload all layers to GPU when one is available.

## Template File

The `.env.example` file provides a ready-to-use template for local development:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=documents
POSTGRES_USER=postgres
POSTGRES_PASSWORD=example
JOBS_TABLE=jobs

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_docs

# RAG Configuration
RAG_DEFAULT_LIMIT=5
RAG_MAX_TOKENS=1000
RAG_SCORE_THRESHOLD=0.35

# RAG Chunking Configuration
RAG_CHUNK_TARGET_WORDS=150
RAG_CHUNK_MAX_WORDS=250
RAG_CHUNK_OVERLAP_WORDS=30

# Worker Identity Configuration
WORKER_ID=
WORKER_NAME=
HEARTBEAT_INTERVAL=15

# Worker Capabilities Configuration
WORKER_DISABLE_LLM=false
WORKER_DISABLE_EMBEDDINGS=false

# Task Filtering Configuration
WORKER_ENABLED_TASKS=
WORKER_DISABLED_TASKS=

# Background Processing Hours (24h format)
BACKGROUND_HOURS_START=2
BACKGROUND_HOURS_END=6

# File Storage Configuration
DOCUMENTS_STORAGE_DIR=../documents
```
