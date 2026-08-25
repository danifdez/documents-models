# Documents models

> WARNING: This project is in ALPHA — features are experimental and may change without notice. Use at your own risk.

## Overview

The models service is the AI/ML running layer of the [documents](https://github.com/danifdez/documents-dev) project.
It runs as a background worker that receives fenced step assignments from Documents Backend over an authenticated
HTTP protocol, processes documents using AI and NLP models, and retries result delivery until Backend acknowledges it.

It is designed to run alongside the backend service and can be deployed on any machine — including CPU-only,
GPU-accelerated, or multi-worker setups. Workers report hardware and task capabilities so Backend only assigns
compatible steps.

## What it does

### Document running

- **Extraction** — Converts uploaded files (PDF, DOC/DOCX, HTML, plain text, ODT, EML, audio/video) into
  clean, normalized HTML. Audio and video files return a metadata summary card.
- **Language detection** — Identifies the language of a document or text sample.
- **Summarization** — Generates concise summaries with cross-lingual support (source and target language can differ).
- **Translation** — Translates text between language pairs using Helsinki-NLP OPUS models.
- **Entity extraction** — Detects people, organizations, locations and other named entities using the local Qwen LLM (multilingual, GBNF-constrained JSON output), extracting entities in the document's original language.
- **Keyword extraction** — Extracts the most relevant keywords and topic phrases from a document.
- **Key point extraction** — Produces a short list of key takeaways from long documents.
- **Dataset statistics** — Computes descriptive statistics (mean, std, top values, etc.) for structured datasets.

### Semantic search and RAG

- **Ingestion** — Chunks document content and returns vector embeddings for Backend-owned persistence.
- **Semantic search** — Finds the most relevant document fragments for a given query using cosine similarity.
- **Question answering (RAG)** — Ranks the supplied context and generates grounded answers using the configured GGUF model.

### Infrastructure

- **Backend-owned scheduling** — Backend applies priority, dependency, deadline and retry policy before granting a step.
- **Multi-worker support** — Multiple instances can run on different machines and receive compatible work from Backend.
- **Hardware detection** — At startup each worker detects CPU cores, RAM, GPU and VRAM, and registers
  its capabilities. Workers without a GPU or LLM skip executions that require them.
- **Fenced attempts** — Every assignment carries a lease-bound attempt identity; Backend rejects stale results.
- **Heartbeat & recovery** — Workers heartbeat through Backend, persist undelivered results locally and resend them until a terminal ACK.

## Models used

| Capability | Model |
|------------|-------|
| Embeddings | `intfloat/multilingual-e5-small` (384-dim, sentence-transformers) |
| Summarization | `facebook/mbart-large-50-one-to-many-mmt` |
| Translation | `Helsinki-NLP/opus-mt-{src}-{tgt}` (per language pair) |
| NER | Local Qwen LLM (multilingual, GBNF-constrained JSON), model configured in `tasks.json` |
| LLM (keywords, key points, Q&A) | GGUF model (configured in `tasks.json`, default: Qwen3-8B) |

All models are downloaded automatically: most from Hugging Face on first use, and the base GGUF LLM
is pre-fetched by the `install` script (and by `setup_models.py --setup` in the standalone bundle)
into the `models/` directory, using the filename declared in `config/tasks.json`.

LLM tasks optionally support LoRA adapters (`lora_model`, `lora_scale` in the task entry) applied on top
of the base GGUF. Adapter files are placed manually. See [docs/configuration.md](docs/configuration.md#tasksjson).

## Requirements

- Python 3.11+
- Access to Documents Backend's authenticated worker protocol
- Docker (optional, recommended)

GPU acceleration (CUDA 12.6) is supported but not required. CPU-only workers handle all tasks except
those that explicitly need a GPU.

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Tasks](docs/tasks.md)
- [Execution types](docs/execution-types.md)
- [NLP tasks](docs/nlp-tasks.md)
- [RAG pipeline](docs/rag-pipeline.md)
- [Document extraction](docs/document-extraction.md)
- [Data storage](docs/database.md)
- [Creating tasks](docs/creating-tasks.md)

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.
