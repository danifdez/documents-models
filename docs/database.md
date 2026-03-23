# Data Storage

The models service uses two databases: **PostgreSQL** for job queue and worker management, and **Qdrant** for vector storage (semantic search and RAG).

## PostgreSQL — Job Queue

### Jobs Table

The `jobs` table (configurable via `JOBS_TABLE` env var) stores all processing requests and their results.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Job identifier |
| `type` | string | Job type (e.g., `summarize`, `search`) |
| `payload` | JSON | Input data for the handler |
| `status` | string | Current state: `pending`, `processing`, `processed`, or `failed` |
| `result` | JSON | Handler output (populated after processing) |
| `priority` | string | Execution priority: `high`, `normal`, or `background` |
| `claimed_by` | UUID | Worker ID that claimed this job (set on `processing`) |
| `started_at` | timestamp | When the job was claimed by a worker |
| `retry_count` | integer | Number of times the job has been requeued after a worker failure |
| `created_at` | timestamp | When the job was created (used for FIFO ordering) |

### Status Lifecycle

```
pending ──> processing ──> processed
                      └──> failed
```

1. **pending** — Job created by the backend, waiting to be picked up.
2. **processing** — A worker has atomically claimed this job and is executing the handler.
3. **processed** — Handler completed successfully and the result has been written.
4. **failed** — Handler raised an exception, or the job exceeded the maximum retry count after worker failures.

### Priority Ordering

Jobs are claimed in strict priority order using `SELECT FOR UPDATE SKIP LOCKED`:

1. `high` — Interactive queries, processed first
2. `normal` — Standard background processing
3. `background` — Only eligible when no `high`/`normal` jobs are pending, or during the configured off-peak window

Within the same priority level, jobs are ordered by `created_at ASC` (FIFO).

### Workers Table

The `workers` table tracks all registered worker instances.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Worker identifier (stable across restarts) |
| `name` | string | Human-readable worker name |
| `capabilities` | JSON array | List of capability tags (e.g., `["gpu", "llm", "embeddings"]`) |
| `status` | string | `online` or `offline` |
| `last_heartbeat` | timestamp | Last heartbeat update (used for stale detection) |
| `started_at` | timestamp | When this worker instance started |
| `metadata` | JSON | Hardware info: `cpu_count`, `ram_gb`, `has_cuda`, `gpu_name`, `vram_gb` |

Workers register on startup and mark themselves `offline` on graceful shutdown. If a worker disappears without shutting down cleanly, its `last_heartbeat` goes stale, and other workers will requeue any jobs it was processing.

### Connection

The `Job` class (`database/job.py`) connects to PostgreSQL using `psycopg` with:

- **Autocommit** enabled for the main connection (read and status-update queries)
- A separate non-autocommit connection is used inside `claim_pending_job()` to wrap the `SELECT FOR UPDATE` + `UPDATE` in a single transaction
- **Dict rows** (`dict_row` factory) — query results are returned as dictionaries

A singleton instance is shared across the application via `get_job_database()`.

### Operations

| Method | Description |
|--------|-------------|
| `claim_pending_job(worker_id, capabilities)` | Atomically claim the highest-priority eligible pending job using `SELECT FOR UPDATE SKIP LOCKED` |
| `requeue_stale_jobs(timeout_seconds, max_retries)` | Reset `processing` jobs from dead workers to `pending` (or `failed` if retries exhausted) |
| `update_job_status(job_id, status)` | Set job status (`processing`, `processed`, `failed`) |
| `update_job_result(job_id, result)` | Write the handler's result dict as JSON |
| `get_connection()` | Return a new independent database connection |

## Qdrant — Vector Database

### Collection: `rag_docs`

The default Qdrant collection (configurable via `QDRANT_COLLECTION` env var) stores document embeddings for the RAG pipeline.

| Property | Value |
|----------|-------|
| **Vector size** | 384 (BAAI/bge-small-en-v1.5 output dimension) |
| **Distance metric** | Cosine similarity |
| **Auto-creation** | Collection is created automatically if it does not exist |
| **Payload indexes** | `project_id` (keyword), `source_id` (keyword) — created automatically for efficient filtering |

### Point Payload

Each point (vector + metadata) in the collection has the following payload structure:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The text chunk that was embedded |
| `source_id` | string | Identifier for the source: numeric resource ID, `doc_{id}`, or `knowledge_{id}` |
| `source_type` | string | Type of source: `resource`, `doc`, or `knowledge` |
| `project_id` | string | UUID of the project the resource belongs to |
| `part_number` | integer | Sequential chunk number within the source document (1-based) |
| `total_chunks` | integer | Total number of chunks for this source |

Point IDs are random UUIDs generated during ingestion.

### Connection

The `Rag` class (`database/rag.py`) connects to Qdrant using `qdrant-client`:

- URL is constructed from `QDRANT_HOST` and `QDRANT_PORT`, or set directly via `QDRANT_URL`.
- On initialization, the collection and its payload indexes are created if they do not exist.

A singleton instance is shared across the application via `get_rag()`.

### Operations

| Method | Description |
|--------|-------------|
| `upsert_points(points)` | Insert or update points in the collection |
| `query_points(query_vector, limit, with_payload, project_id, score_threshold)` | Find similar vectors; optionally filter by project and minimum score |
| `delete_by_source(source_id)` | Remove all points belonging to a source (used before re-ingesting) |
| `delete_points(point_ids)` | Remove points by their IDs |
| `get_collection_info()` | Retrieve collection metadata and statistics |
| `recreate_collection()` | Drop and recreate the collection (use for schema migrations) |
