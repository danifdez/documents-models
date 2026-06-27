# Getting Started

## Prerequisites

### Python

- **Python 3.11+** is required.

### System Dependencies

The following system packages are needed for compiling native extensions (llama-cpp-python, Docling):

```bash
build-essential cmake ninja-build python3-dev git
libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

On Debian/Ubuntu:

```bash
sudo apt-get install build-essential cmake ninja-build python3-dev git \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

## Installation

Use the provided install script in the `models` directory to create the virtualenv and install dependencies:

```bash
cd models
chmod +x install && ./install
```

The script creates `config/tasks.json` from defaults (prompting for database, Qdrant, and storage settings), sets up the venv, and installs CPU/GPU dependencies (if CUDA is detected). After it finishes you can activate the virtualenv with `source .venv/bin/activate` and start the worker with `python jobs.py`.

## Running the service

Run locally using the created virtual environment:

```bash
source .venv/bin/activate
python jobs.py
```

## AI Models

The service uses several AI models. Most are downloaded automatically from Hugging Face on first use. Which model is used for each task is controlled by `config/tasks.json` (auto-created from `common/tasks.default.json`).

| Model | Purpose | Downloaded By |
|-------|---------|---------------|
| `intfloat/multilingual-e5-small` | Text embeddings (384-dim, multilingual) | sentence-transformers (auto on first request) |
| `facebook/mbart-large-50-one-to-many-mmt` | Summarization | Hugging Face transformers (auto on first request) |
| `Helsinki-NLP/opus-mt-{src}-{tgt}` | Translation (per language pair) | Hugging Face transformers (auto on first request) |
| GGUF LLM (see `tasks.json`, default: Qwen3-8B) | Key points, keywords, Q&A | Auto-downloaded from `Qwen/Qwen3-8B-GGUF` into `models/` by `install` / `setup_models.py` |

### LLM Setup

The base LLM GGUF is auto-downloaded into the `models/` subdirectory: by the `install` script during local setup, and by `setup_models.py --setup` when the standalone bundle is installed by the Electron app. The filename used is whatever is declared in `config/tasks.json` under the `keywords`, `key-point`, `ask`, and `summarize` task entries (default: `Qwen3-8B-Q5_K_M.gguf`). If the file is missing at runtime, tasks fall back to heuristics or fail gracefully.

**Optional LoRA adapters.** To fine-tune any LLM task, place a LoRA adapter `.gguf` in `models/` and add `lora_model` (and optionally `lora_scale`) to the task entry. See [configuration.md](configuration.md#tasksjson) for details.

## Verifying the Service

Once started, the service logs hardware info, registers the worker, and then prints:

```
Worker registered: <name> (<id>)
Capabilities: ['llm', 'embeddings']
Job service started. Polling for pending jobs...
```

It polls the PostgreSQL `jobs` table every 1 second for pending jobs. To verify processing:

1. Insert a test job into the `jobs` table with `status = 'pending'` and a valid `type` (e.g., `detect-language`).
2. Watch the service logs for `Processing job: <id> of type: <type>`.
3. Check the `jobs` table — the row should have `status = 'processed'` and a populated `result` column.
