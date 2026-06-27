# Data Storage

The models service uses **PostgreSQL** for everything: the job queue and worker management, and — via the `vector` (pgvector) extension — vector storage for semantic search and RAG.

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

## Vector Storage — pgvector

Document embeddings live in PostgreSQL via the `vector` (pgvector) extension — there is no separate vector database. All vectors are E5 multilingual embeddings (384 dimensions, cosine distance, output of intfloat/multilingual-e5-small).

The extension and the tables below are created by a backend TypeORM migration (`CreateVectorTables`), following the migration-first schema rule. The worker only reads and writes them. There are three tables, one per domain, kept physically separate for isolation:

| Table | Scope | Cleanup / filtering |
|-------|-------|---------------------|
| `rag_chunks` | Workspace RAG (resources, docs, knowledge). No foreign key, since sources are heterogeneous. | Deleted by `source_id` |
| `indexed_file_chunks` | Files in the assistant's working folder. FK to `indexed_files` (`ON DELETE CASCADE`). | Filtered by the `owner_tag` column |
| `memory_vectors` | Assistant memory, 1-to-1 with `assistant_memory_entries` (FK + `CASCADE`). PK is `memory_id`. | Upserted in place per `memory_id` |

### `rag_chunks` columns

| Column | Type | Description |
|-------|------|-------------|
| `id` | UUID | Random UUID generated during ingestion |
| `vector` | `vector(384)` | The chunk embedding |
| `text` | string | The text chunk that was embedded |
| `source_id` | string | Identifier for the source: numeric resource ID, `doc_{id}`, or `knowledge_{id}` |
| `source_type` | string | Type of source: `resource`, `doc`, or `knowledge` |
| `project_id` | string | UUID of the project the resource belongs to |
| `part_number` | integer | Sequential chunk number within the source document (1-based) |
| `total_chunks` | integer | Total number of chunks for this source |

### Connection

The `Rag` class (`database/rag.py`) reads and writes these tables with `psycopg` and the `pgvector` package, reusing the PostgreSQL connection. Table names come from the `vectors` block in `config/config.json`.

A singleton instance is shared across the application via `get_rag()`.

### Operations

| Method | Description |
|--------|-------------|
| `upsert_points(points)` | Insert or update vector rows |
| `query_points(query_vector, limit, with_payload, project_id, score_threshold)` | Find similar vectors (pgvector cosine); optionally filter by project and minimum score |
| `delete_by_source(source_id)` | Remove all rows belonging to a source (used before re-ingesting) |
| `delete_points(point_ids)` | Remove rows by their IDs |
