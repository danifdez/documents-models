# Architecture

## System Overview

```
+------------------+       creates jobs       +------------------+
|                  | -----------------------> |                  |
|  NestJS Backend  |                          |    PostgreSQL    |
|   (Port 3000)    | <--- reads results ----- |   jobs + workers |
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
                |  File Storage   |          |      Qdrant       |
                | /app/documents  |          |  Vector Database  |
                |    _storage     |          |   (Port 6333)     |
                +-----------------+          +-------------------+
```

The models service is a background worker that sits between the backend and the AI/ML infrastructure. The backend creates jobs in the PostgreSQL `jobs` table. One or more worker instances poll for pending jobs, claim them atomically, process them using the appropriate AI model, and write results back. Multiple workers can run simultaneously on different machines and share load automatically.

## Application Bootstrap

The entry point is `jobs.py`:

1. Detects hardware (CPU count, RAM, GPU/CUDA/VRAM) and determines worker capabilities.
2. Registers the worker in the `workers` table (or updates an existing row on restart).
3. Starts a background heartbeat thread (default: every 15 seconds).
4. Registers SIGTERM/SIGINT handlers for graceful shutdown.
5. Enters an infinite polling loop (1-second interval):
   - Requeues jobs from dead workers (`requeue_stale_jobs()`).
   - Atomically claims and processes one pending job (`claim_pending_job()`).

```python
# Simplified flow
capabilities = detect_worker_capabilities()
register_worker(capabilities, metadata)
start_heartbeat_thread()

db = get_job_database()
while True:
    db.requeue_stale_jobs()
    job = db.claim_pending_job(WORKER_ID, capabilities)
    if job:
        process_job(job)
    time.sleep(1)
```

## Directory Structure

```
models/
├── config.py                    # Configuration constants (reads from config/config.json)
├── jobs.py                      # Entry point — worker bootstrap and polling loop
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── common/
│   ├── config.default.json      # Default general configuration
│   └── tasks.default.json       # Default task configuration
├── config/                      # User configuration (created by install, .gitignored)
│   ├── config.json              # General settings (DB, Qdrant, storage, worker, RAG, LLM)
│   ├── tasks.json               # Task settings (models, capabilities, parameters)
│   └── tasks/                   # Per-task overrides (prompt.md, config.json)
├── database/
│   ├── job.py                  # PostgreSQL job queue + worker operations (Job class)
│   └── rag.py                  # Qdrant vector database operations (Rag class)
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
│   ├── llm_service.py          # llama-cpp-python LLM wrapper (cached instances)
│   ├── model_config.py         # Configuration loader (config.json + tasks.json + overrides)
│   ├── prompts.py              # Prompt loader (config/tasks/ -> tasks/<dir>/prompt.md)
│   └── text.py                 # HTML text extraction and semantic chunking
├── tasks/
│   ├── base.py                  # Task interface definition and TaskDefinition dataclass
│   ├── ask/                    # RAG question answering (+ prompt.md)
│   ├── dataset_stats/          # Dataset statistics computation
│   ├── detect_language/        # Language detection
│   ├── embedding/              # Text-to-vector conversion
│   ├── entities/               # Named entity extraction (spaCy)
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
│   ├── job_registry.py         # @job_handler decorator and handler registry
│   └── process_job.py          # Job dispatch and lifecycle management
└── worker/
    ├── capabilities.py         # Capability detection (reads task requirements from JSON)
    └── identity.py             # Worker ID, name, registration and heartbeat
```

## Job Lifecycle

```
pending ──> processing ──> processed
                      └──> failed
```

1. The backend inserts a job with `status = 'pending'` and a `priority` (`high`, `normal`, or `background`).
2. `claim_pending_job()` uses `SELECT FOR UPDATE SKIP LOCKED` to atomically claim the highest-priority job the worker can handle. The job is updated to `status = 'processing'`, `claimed_by = worker_id`, and `started_at = NOW()`.
3. The registered handler function executes with the job's `payload`.
4. On success: the handler's return value is stored in `result` and status becomes `processed`.
5. On failure: the status becomes `failed`.
6. If a worker dies mid-job, `requeue_stale_jobs()` detects the stale heartbeat and resets the job to `pending` (up to `max_retries = 3` times); after that the job is marked `failed`.

## Key Design Patterns

### Decorator-Based Job Registration

Task handlers register themselves using the `@job_handler` decorator:

```python
from utils.job_registry import job_handler

@job_handler("summarize")
def summarize_text(payload) -> dict:
    # ...
```

The decorator adds the function to a global `TASK_HANDLERS` dictionary keyed by job type. At startup, `process_job.py` imports all task modules, which triggers registration. The dispatcher looks up the handler by `job["type"]`.

### Capability-Based Job Routing

Each task type declares the worker capabilities it requires (`worker/capabilities.py`). Workers detect their own capabilities at startup (GPU, LLM, embeddings) and only claim jobs whose requirements are satisfied. Additional fine-grained control is available via `WORKER_ENABLED_TASKS` and `WORKER_DISABLED_TASKS` environment variables.

### Singleton Services

Database connections and model instances are created once and reused:

- `get_job_database()` — PostgreSQL connection (autocommit, dict rows)
- `get_rag()` — Qdrant client (ensures collection and indexes exist on init)
- `get_embedding_service()` — SentenceTransformer model (loaded on first call)
- `get_llm_service()` — LLM instance, cached per model path

### Per-Task Model Configuration

Model selection is driven by `config/tasks.json` (auto-created from `common/tasks.default.json` on first run). This allows changing the model for a task (e.g., switching from Mistral to Phi-4) without modifying code.

### Modular RAG Pipeline

The RAG system (`rag/`) is composed of independent stage objects each implementing a `.run(ctx: RAGContext)` method:

```
Retriever → Reranker → ContextBuilder → PromptBuilder → Generator
```

`RAGContext` is a mutable dataclass passed through each stage, accumulating results. This makes it easy to add, remove, or reorder stages.

### Graceful Degradation

LLM-dependent tasks (`key-point`, `keywords`) fall back to heuristic extraction when no LLM is available. The `keywords` task also handles environments where `llama-cpp-python` is not installed.

### Format-Agnostic Extraction

All document formats (PDF, DOC/DOCX, HTML, TXT) are normalized to clean HTML output — no inline styles, classes, or IDs. This gives downstream tasks a uniform input format regardless of the original document type.

### Atomic Priority Queue

Jobs are claimed in strict priority order (`high > normal > background`) using `SELECT FOR UPDATE SKIP LOCKED`. Background jobs are only eligible when no high/normal jobs are pending, or during the configured off-peak window (`BACKGROUND_HOURS_START`–`BACKGROUND_HOURS_END`).
