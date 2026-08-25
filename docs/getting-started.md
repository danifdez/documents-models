# Getting Started

## Prerequisites

### Python

- **Python 3.11+** is required.

### System Dependencies

The following system packages are needed for compiling native extensions (Docling):

```bash
build-essential cmake ninja-build python3-dev git
libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

On Debian/Ubuntu:

```bash
sudo apt-get install build-essential cmake ninja-build python3-dev git \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### Inference engine

Text generation does not happen in this process. It happens in a llama-server that
Models installs (`bash install`, into `models/bin/llama/`) and runs as
one more service (`bash manage start llama`), and that every worker — plus the
embedded browser — talks to over HTTP. There is nothing to compile here: the engine
is a binary, and whether it uses the GPU depends on which build is installed and on
`llm_defaults.n_gpu_layers`.

Running this service on its own, without `manage`:

```bash
source .venv/bin/activate
python -m services.llama_server     # starts the engine in the foreground
BACKEND_URL=http://localhost:3000 \
MODELS_ENROLLMENT_TOKEN=<backend-token> \
python executions.py                # registers with Backend and claims steps
```

## Installation

Use the provided install script in the `models` directory to create the virtualenv and install dependencies:

```bash
cd models
chmod +x install && ./install
```

The script creates `config/tasks.json` from defaults, prompts for task-domain database and storage settings, sets up the venv, and installs CPU/GPU dependencies (if CUDA is detected). After it finishes you can activate the virtualenv and start the worker through the root `manage` command or with the standalone environment shown above.

## Running the service

Run locally using the created virtual environment:

```bash
source .venv/bin/activate
python executions.py
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

The base LLM GGUF is auto-downloaded into the `models/` subdirectory: by the `install` script during local setup, and by `setup_models.py --setup` when the standalone bundle is installed by the Electron app. The filename used is whatever is declared in `config/tasks.json` under the `keywords`, `key-point`, `ask`, `summarize-map`, and `summarize-reduce` task entries (default: `Qwen3-8B-Q5_K_M.gguf`). If the file is missing at runtime, tasks fall back to heuristics or fail gracefully.

**Optional LoRA adapters.** To fine-tune any LLM task, place a LoRA adapter `.gguf` in `models/` and add `lora_model` (and optionally `lora_scale`) to the task entry. See [configuration.md](configuration.md#tasksjson) for details.

## Verifying the Service

Once started, the service logs hardware info and registers the worker through Backend:

```
Worker registered through Backend: <name> (<id>)
```

To verify the service:

1. Start Backend and Models with the same `MODELS_ENROLLMENT_TOKEN`.
2. Create a supported execution through the Backend API or application UI.
3. Confirm that Models registers and Backend grants a compatible step.
4. Confirm that Models logs the terminal result ACK. Models must never read or update execution control-plane tables directly.
