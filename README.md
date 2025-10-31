# documents-models

## Overview

This service is part of the documents project and provides Python-based microservices for document processing.

## Features

- **Job Polling & Processing:** Checks for new jobs, processes them, and updates job and resource status.
- **Document Extraction:** Handles HTML, PDF, DOC, TXT, outputting normalized HTML/content.
- **Language Detection:** Identifies the language of extracted document content for downstream processing.
- **Summarization:** Generates concise summaries of documents using AI models.
- **Translation:** Translates document content between supported languages.
- **Search:** Performs semantic and keyword-based search using vector embeddings.
- **Ingest:** Normalizes and stores processed documents and metadata for later retrieval.
- **Entities:** Extracts named entities (people, organizations, locations, etc.) from documents.
- **RAG (Retrieval-Augmented Generation):** Combines document retrieval with generative models for advanced question answering.

## Installation

You can run the models service either with Docker or manually:

### Docker

1. Build the Docker image for the models service:

   ```bash
   docker build -t documents-models .
   ```

2. Run the container (set environment variables as needed for PostgreSQL/Qdrant):

   ```bash
   docker run --rm \
     -e POSTGRES_HOST="<host>" \
     -e POSTGRES_PORT="<port>" \
     -e POSTGRES_DB="<database>" \
     -e POSTGRES_USER="<user>" \
     -e POSTGRES_PASSWORD="<password>" \
     -e QDRANT_HOST="<host>" \
     -e QDRANT_PORT="<port>" \
     -e QDRANT_COLLECTION="<collection>" \
     -e LLM_MODEL_NAME="Mistral-7B-Instruct-v0.3-Q8_0.gguf" \
     documents-models
   ```

   The service will start and begin polling for jobs. Adjust environment variables as needed for your setup.

## Configuration

The models service can be configured using environment variables or by modifying the `config.py` file. Available configuration options:

### LLM Configuration

- `LLM_MODEL_NAME`: Name of the LLM model file in `/app/models/` directory (default: `Mistral-7B-Instruct-v0.3-Q8_0.gguf`)
- `LLM_N_CTX`: Context window size for the LLM (default: `32768`)
- `LLM_N_THREADS`: Number of threads for LLM processing (default: `4`)
- `LLM_N_BATCH`: Batch size for LLM processing (default: `64`)

### Embedding Configuration

- `EMBEDDING_MODEL_NAME`: Name of the embedding model to use (default: `BAAI/bge-small-en-v1.5`)

### RAG Configuration

- `RAG_DEFAULT_LIMIT`: Default number of results to retrieve from vector database (default: `5`)
- `RAG_MAX_TOKENS`: Maximum tokens for generated responses (default: `1000`)

Example with custom configuration:

```bash
docker run --rm \
  -e LLM_MODEL_NAME="custom-model.gguf" \
  -e LLM_N_CTX="16384" \
  -e LLM_N_THREADS="8" \
  documents-models
```

### Local

1. Install system dependencies:

   - Python 3.11 or newer
   - pip (Python package manager)

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the models service:

   ```bash
   python jobs.py
   ```

   The service will begin polling for jobs and processing them.

## Getting Started

To manually create jobs for processing, create rows in your jobs table (PostgreSQL) compatible with the models service. Example SQL to create a compatible jobs table:

```sql
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  type TEXT,
  payload JSONB,
  status TEXT,
  priority TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  result JSONB
);
```

Example job insert (psql):

```sql
INSERT INTO jobs (id, type, payload, status, priority)
VALUES ('job-123', 'summarize', '{"content": "..."}', 'pending', 'normal');
```

### Example Job Structures

#### Summarize

```json
{
  "_id": "<job_id>",
  "type": "summarize",
  "payload": {
    "content": "<document text>",
    "sourceLanguage": "en",
    "targetLanguage": "es"
  }
}
```

**Result:**

```json
{
  "summary": "<summary text>"
}
```

#### Detect Language

```json
{
  "_id": "<job_id>",
  "type": "detect-language",
  "payload": {
    "samples": ["Text sample 1", "Text sample 2"]
  }
}
```

**Result:**

```json
{
  "results": [
    { "text": "Text sample 1", "language": "en" },
    { "text": "Text sample 2", "language": "es" }
  ]
}
```

#### Entities

```json
{
  "_id": "<job_id>",
  "type": "entity-extraction",
  "payload": {
    "texts": [{ "text": "John Doe works at Acme Corp." }]
  }
}
```

**Result:**

```json
{
  "entities": [
    { "type": "PERSON", "text": "John Doe" },
    { "type": "ORG", "text": "Acme Corp" }
  ]
}
```

#### Extraction

```json
{
  "_id": "<job_id>",
  "type": "document-extraction",
  "payload": {
    "hash": "<file_hash>",
    "extension": ".pdf"
  }
}
```

**Result:**

```json
{
  "content": "<normalized HTML>",
  "metadata": { "title": "<title>", "author": "<author>" }
}
```

#### Ingest

```json
{
  "_id": "<job_id>",
  "type": "ingest-content",
  "payload": {
    "source_id": "<source_id>",
    "project_id": "<project_id>",
    "content": "<normalized HTML>"
  }
}
```

**Result:**

```json
{
  "success": true
}
```

#### Translate

```json
{
  "_id": "<job_id>",
  "type": "translate",
  "payload": {
    "texts": [{ "text": "Texto a traducir" }],
    "sourceLanguage": "es",
    "targetLanguage": "en"
  }
}
```

**Result:**

```json
{
  "translated_texts": [{ "translation_text": "Text to translate" }]
}
```

#### Ask

```json
{
  "_id": "<job_id>",
  "type": "ask",
  "payload": {
    "question": "<your question>"
  }
}
```

**Result:**

```json
{
  "answer": "<answer to your question>"
}
```

#### Search

```json
{
  "_id": "<job_id>",
  "type": "search",
  "payload": {
    "query": "<search query>",
    "limit": 5
  }
}
```

**Result:**

```json
{
  "results": [
    { "text": "...", "score": 0.92, "metadata": {} },
    { "text": "...", "score": 0.87, "metadata": {} }
  ]
}
```

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.
