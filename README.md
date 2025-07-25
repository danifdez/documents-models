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
2. Run the container (set environment variables as needed for MongoDB/Qdrant):
   ```bash
   docker run --rm \
     -e MONGO_URI="mongodb://<host>:<port>/<db>" \
     -e MONGO_DB_NAME="<database>" \
     -e QDRANT_HOST="<host>" \
     -e QDRANT_PORT="<port>" \
     -e QDRANT_COLLECTION="<collection>" \
     documents-models
   ```
   The service will start and begin polling for jobs. Adjust environment variables as needed for your setup.

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

To manually create jobs in MongoDB for processing, use the following example structures for each job type:

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
