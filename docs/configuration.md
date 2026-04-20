# Configuration

Configuration is split into two JSON files inside `config/`:

- **`config/config.json`** — General settings (database, Qdrant, storage, LLM defaults, RAG, worker)
- **`config/tasks.json`** — Per-task configuration (models, capabilities, parameters, enabled/disabled)

Both are auto-created from `common/config.default.json` and `common/tasks.default.json` during installation. Edit them directly to change any setting.

Per-task overrides can also be placed in `config/tasks/<task-name>/`.

## config.json

### Database

```json
"database": {
  "host": "localhost",
  "port": 5432,
  "name": "documents",
  "user": "postgres",
  "password": "example",
  "jobs_table": "jobs"
}
```

### Qdrant

```json
"qdrant": {
  "enabled": true,
  "host": "localhost",
  "port": 6333,
  "collection": "rag_docs"
}
```

Set `enabled` to `false` to disable all RAG/embedding features. This automatically disables tasks that require embeddings.

### Storage

```json
"storage": {
  "documents_dir": "../documents"
}
```

### LLM Defaults

Shared parameters for all LLM-based tasks. Individual tasks can override any of these.

```json
"llm_defaults": {
  "model_dir": "models",
  "n_ctx": 32768,
  "n_threads": 4,
  "n_batch": 64,
  "n_gpu_layers": 0
}
```

> **Note:** Set `n_gpu_layers` to `-1` to offload all layers to GPU when one is available.

### RAG

```json
"rag": {
  "default_limit": 5,
  "max_tokens": 1000,
  "score_threshold": 0.35,
  "chunk_target_words": 150,
  "chunk_max_words": 250,
  "chunk_overlap_words": 30
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `default_limit` | `5` | Number of chunks retrieved for the `ask` task |
| `max_tokens` | `1000` | Maximum tokens in LLM responses for RAG |
| `score_threshold` | `0.35` | Minimum cosine similarity to include a chunk |
| `chunk_target_words` | `150` | Target chunk size in words |
| `chunk_max_words` | `250` | Maximum words before a chunk is split |
| `chunk_overlap_words` | `30` | Overlap words between consecutive chunks |

### Worker

```json
"worker": {
  "id": "",
  "name": "",
  "heartbeat_interval": 15,
  "disable_llm": false,
  "disable_embeddings": false,
  "background_hours_start": 2,
  "background_hours_end": 6
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `id` | auto UUID (persisted to `.worker_id`) | Stable worker identity across restarts |
| `name` | `worker-{id[:8]}` | Human-readable name for logs |
| `heartbeat_interval` | `15` | Seconds between heartbeat updates |
| `disable_llm` | `false` | Disable all LLM capabilities on this worker |
| `disable_embeddings` | `false` | Disable all embedding capabilities on this worker |
| `background_hours_start` | `2` | Hour (0-23) when background window opens |
| `background_hours_end` | `6` | Hour (0-23) when background window closes |

`background` priority jobs are processed only when no `high`/`normal` jobs are pending, **or** when the current time falls inside the background window.

## tasks.json

Each task has its own entry as a top-level key:

```json
{
  "keywords": {
    "enabled": true,
    "type": "llm",
    "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
    "capabilities": ["llm"],
    "max_tokens": 500,
    "max_items": 10,
    "max_words_per_item": 3
  }
}
```

Common fields for all tasks:

| Field | Required | Description |
|-------|----------|-------------|
| `enabled` | yes | `true`/`false` — whether this task is active |
| `type` | yes | Model type: `llm`, `sentence-transformer`, `seq2seq`, `spacy`, `translation`, `rag`, `pipeline`, `utility` |
| `capabilities` | yes | Required worker capabilities (`["llm"]`, `["embeddings"]`, `["llm", "embeddings"]`, or `[]`) |
| `model` | no | Model name or path |

Optional fields for `type: "llm"` tasks (applied on top of the base GGUF model):

| Field | Default | Description |
|-------|---------|-------------|
| `lora_model` | — | Filename of a LoRA adapter `.gguf` inside `model_dir`. Place the file manually. |
| `lora_path` | — | Absolute path to the LoRA adapter. Overrides `lora_model` when set. |
| `lora_scale` | `1.0` | Blend scale for the LoRA adapter. |

Each `(model, lora_path, lora_scale)` combination is cached as a separate Llama instance, so different tasks can use different base+adapter pairs without collision.

Additional task-specific parameters vary by task (see `common/tasks.default.json` for the full default configuration).

## Per-task Overrides

You can override task behavior without editing `config/tasks.json`:

### Prompt Override

Create `config/tasks/<task-type>/prompt.md` with your custom prompt. This takes priority over the default prompt in `tasks/<task-dir>/prompt.md`.

### Config Override

Create `config/tasks/<task-type>/config.json` with parameter overrides. These are merged on top of the task's entry in `tasks.json`:

```json
{
  "max_tokens": 2000,
  "max_items": 20
}
```

## Installation

Run `./install` to create `config/config.json` and `config/tasks.json` interactively. The script prompts for database, Qdrant (optional), and storage settings.

See [creating-tasks.md](creating-tasks.md) for how to add new tasks.
