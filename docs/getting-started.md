# Getting Started

## Prerequisites

### Python

- **Python 3.11+** is required.

### System Dependencies

The following system packages are needed for compiling native extensions (llama-cpp-python, Docling, spaCy):

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

The script creates `config/tasks.json` from defaults (prompting for database, Qdrant, and storage settings), sets up the venv, installs CPU/GPU dependencies (if CUDA is detected), and downloads spaCy models. After it finishes you can activate the virtualenv with `source .venv/bin/activate` and start the worker with `python jobs.py`.

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
| spaCy model (see `tasks.json`) | NER | `python -m spacy download <model>` (Docker build runs this for the default model) |
| `BAAI/bge-small-en-v1.5` | Text embeddings (384-dim) | sentence-transformers (auto on first request) |
| `facebook/mbart-large-50-one-to-many-mmt` | Summarization | Hugging Face transformers (auto on first request) |
| `Helsinki-NLP/opus-mt-{src}-{tgt}` | Translation (per language pair) | Hugging Face transformers (auto on first request) |
| GGUF LLM (see `tasks.json`, default: Phi-4-mini-instruct) | Key points, keywords, Q&A | Must be placed manually in the `models/` directory |

### LLM Setup

The LLM (GGUF file) is **not** auto-downloaded. Place the model file in the `models/` subdirectory, then set the filename in `config/tasks.json` under the `keywords`, `key-point`, and `ask` task entries. Tasks that depend on it will fall back to heuristics or fail gracefully if the file is not present.

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
