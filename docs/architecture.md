# Architecture

## System Overview

```
+------------------+       creates executions       +------------------+
|                  | -----------------------> |                  |
|  NestJS Backend  |                          |    PostgreSQL    |
|   (Port 3000)    | <--- reads results ----- |   executions + workers |
|                  |                          |      tables      |
+------------------+                          +--------+---------+
                                                       |
                                              polls every 1s
                                                       |
                                     +-----------------+-----------------+
                                     |                 |                 |
                            +--------v-------+ +-------v--------+ +------v---------+
                            | Models Worker  | | Models Worker  | | Models Worker  |
                            |   (GPU node)   | |   (CPU node)   | | (lightweight)  |
                            +--------+-------+ +-------+--------+ +------+---------+
                                     |                 |                 |
                         +-----------+-------+---------+-------+---------+
                         |                             |
                +--------v--------+          +---------v---------+
                |  File Storage   |          |    PostgreSQL     |
                | /app/documents  |          | vector tables     |
                |    _storage     |          |    (pgvector)     |
                +-----------------+          +-------------------+
```

The models service is a background worker that sits between the backend and the AI/ML infrastructure. The backend creates executions in the PostgreSQL `executions` table. One or more worker instances poll for queued executions, claim them atomically, process them using the appropriate AI model, and write results back. Multiple workers can run simultaneously on different machines and share load automatically.

## Application Bootstrap

The entry point is `executions.py`:

1. Detects hardware (CPU count, RAM, GPU/CUDA/VRAM) and determines worker capabilities.
2. Registers the worker in the `workers` table (or updates an existing row on restart).
3. Starts a background heartbeat thread (default: every 15 seconds).
4. Registers SIGTERM/SIGINT handlers for graceful shutdown.
5. Enters an infinite polling loop (1-second interval):
   - Requeues executions from dead workers (`requeue_stale_executions()`).
   - Atomically claims and processes one queued execution (`claim_pending_execution()`).

```python
# Simplified flow
capabilities = detect_worker_capabilities()
register_worker(capabilities, metadata)
start_heartbeat_thread()

db = get_execution_database()
while True:
    db.requeue_stale_executions()
    execution = db.claim_pending_execution(WORKER_ID, capabilities)
    if execution:
        process_execution(execution)
    time.sleep(1)
```

## Directory Structure

```
models/
├── config.py                    # Configuration constants (reads from config/config.json)
├── executions.py                      # Entry point — worker bootstrap and polling loop
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── common/
│   ├── config.default.json      # Default general configuration
│   └── tasks.default.json       # Default task configuration
├── config/                      # User configuration (created by install, .gitignored)
│   ├── config.json              # General settings (DB, vector tables, storage, worker, RAG, LLM)
│   ├── tasks.json               # Task settings (models, capabilities, parameters)
│   └── tasks/                   # Per-task overrides (prompt.md, config.json)
├── database/
│   ├── execution.py                  # PostgreSQL execution queue + worker operations (Execution class)
│   └── rag.py                  # pgvector vector storage operations (Rag class)
├── rag/
│   ├── pipeline.py             # RAGPipeline orchestrator
│   ├── retriever.py            # Vector search stage
│   ├── reranker.py             # Deduplication and score-sort stage
│   ├── context_builder.py      # Assembles chunks into context text
│   ├── prompt_builder.py       # Builds the LLM prompt from template
│   ├── generator.py            # LLM inference stage
│   └── types.py                # RAGContext and RetrievedChunk dataclasses
├── services/
│   ├── embedding_service.py    # Sentence-transformers embedding wrapper
│   ├── llm_service.py          # HTTP client for the shared llama-server (cached per model)
│   ├── llama_server.py         # Finds (or starts) the shared engine; one definition of it
│   ├── model_config.py         # Configuration loader (config.json + tasks.json + overrides)
│   ├── prompts.py              # Prompt loader (config/tasks/ -> tasks/<dir>/prompt.md)
│   └── text.py                 # HTML text extraction and semantic chunking
├── tasks/
│   ├── base.py                  # Task interface definition and TaskDefinition dataclass
│   ├── ask/                    # RAG question answering (+ prompt.md)
│   ├── dataset_stats/          # Dataset statistics computation
│   ├── detect_language/        # Language detection
│   ├── embedding/              # Text-to-vector conversion
│   ├── entities/               # Named entity extraction (local Qwen LLM)
│   ├── extraction/             # Document extraction pipeline
│   │   ├── extractor.py        # Format router
│   │   └── processors/         # Per-format processors (PDF, DOC, HTML, TXT)
│   ├── ingest/                 # RAG content ingestion + vector deletion
│   ├── key_points/             # Key point extraction (LLM)
│   ├── keywords/               # Keyword extraction (LLM)
│   ├── search/                 # Semantic search
│   ├── summarize/              # Text summarization (mBART)
│   └── translate/              # Machine translation (OPUS)
├── utils/
│   ├── device.py               # Hardware detection (CPU, RAM, GPU, threads)
│   ├── execution_registry.py   # @execution_handler decorator and handler registry
│   └── process_execution.py          # Execution dispatch and lifecycle management
└── worker/
    ├── capabilities.py         # Capability detection (reads task requirements from JSON)
    └── identity.py             # Worker ID, name, registration and heartbeat
```

## Execution Lifecycle

```
queued ──> running ──> completed
                      └──> failed
```

1. The backend inserts a execution with `status = 'queued'` and a `priority` (`high`, `normal`, or `background`).
2. `claim_pending_execution()` uses `SELECT FOR UPDATE SKIP LOCKED` to atomically claim the highest-priority execution the worker can handle. The execution is updated to `status = 'running'`, `claimed_by = worker_id`, and `started_at = NOW()`.
3. The registered handler function executes with the execution's `payload`.
4. On success: the handler's return value is stored in `result` and status becomes `completed`.
5. On failure: the status becomes `failed`.
6. If a worker dies mid-execution, `requeue_stale_executions()` detects the stale heartbeat and resets the execution to `queued` (up to `max_retries = 3` times); after that the execution is marked `failed`.

## Key Design Patterns

### Decorator-Based Execution Registration

Task handlers register themselves using the `@execution_handler` decorator:

```python
from common.execution_registry import execution_handler

@execution_handler("summarize")
def summarize_text(payload) -> dict:
    # ...
```

The decorator adds the function to a global `TASK_HANDLERS` dictionary keyed by execution type. At startup, `process_execution.py` imports all task modules, which triggers registration. The dispatcher looks up the handler by `execution["type"]`.

### Capability-Based Execution Routing

Each task type declares the worker capabilities it requires (`worker/capabilities.py`). Workers detect their own capabilities at startup (GPU, LLM, embeddings) and only claim executions whose requirements are satisfied. Additional fine-grained control is available via `WORKER_ENABLED_TASKS` and `WORKER_DISABLED_TASKS` environment variables.

### Singleton Services

Database connections and model instances are created once and reused:

- `get_execution_database()` — PostgreSQL connection (autocommit, dict rows)
- `get_rag()` — pgvector storage accessor (reads/writes the vector tables via psycopg)
- `get_embedding_service()` — SentenceTransformer model (loaded on first call)
- `get_llm_service()` — LLM instance, cached per `(model_path, lora_path, lora_scale)` tuple

### Per-Task Model Configuration

Model selection is driven by `config/tasks.json` (auto-created from `common/tasks.default.json` on first run). This allows changing the model for a task (e.g., switching from Qwen3 to Mistral) without modifying code. Each `type: "llm"` task can also declare an optional `lora_model` (+ `lora_scale`) to apply a LoRA adapter on top of its base GGUF; different base+adapter combinations coexist in the cache as distinct instances.

### Modular RAG Pipeline

The RAG system (`rag/`) is composed of independent stage objects each implementing a `.run(ctx: RAGContext)` method:

```
Retriever → Reranker → ContextBuilder → PromptBuilder → Generator
```

`RAGContext` is a mutable dataclass passed through each stage, accumulating results. This makes it easy to add, remove, or reorder stages.

### Graceful Degradation

LLM-dependent tasks (`key-point`, `keywords`) fall back to heuristic extraction when no LLM is available — which now means the shared llama-server is not reachable and could not be started, rather than a missing Python package.

### Format-Agnostic Extraction

All document formats (PDF, DOC/DOCX, HTML, TXT) are normalized to clean HTML output — no inline styles, classes, or IDs. This gives downstream tasks a uniform input format regardless of the original document type.

### Atomic Priority Queue

Executions are claimed in strict priority order (`high > normal > background`) using `SELECT FOR UPDATE SKIP LOCKED`. Background executions are only eligible when no high/normal executions are queued, or during the configured off-peak window (`BACKGROUND_HOURS_START`–`BACKGROUND_HOURS_END`).
