# Job Types

The models service processes 11 job types, each registered via the `@job_handler` decorator. All handlers receive a `payload` dict and return a result dict.

## Overview

| Type | Handler | File | Model / Library |
|------|---------|------|-----------------|
| `document-extraction` | `extract()` | `tasks/extraction/extractor.py` | Docling, Trafilatura |
| `detect-language` | `detect_language()` | `tasks/detect_language/detect_language.py` | langdetect |
| `summarize` | `summarize_text()` | `tasks/summarize/summarize.py` | mBART-50 |
| `translate` | `translate()` | `tasks/translate/translate.py` | Helsinki-NLP/OPUS |
| `entity-extraction` | `entities()` | `tasks/entities/entities.py` | spaCy en_core_web_trf |
| `ingest-content` | `ingest()` | `tasks/ingest/ingest.py` | BAAI/bge-small-en-v1.5 |
| `search` | `search_snippets()` | `tasks/search/search.py` | BAAI/bge-small-en-v1.5 |
| `ask` | `ask_question()` | `tasks/ask/ask.py` | Mistral-7B + BAAI embeddings |
| `key-point` | `key_points()` | `tasks/key_points/key_points.py` | Mistral-7B (with heuristic fallback) |
| `keywords` | `keywords()` | `tasks/keywords/keywords.py` | Mistral-7B (with heuristic fallback) |
| `embedding` | `create_embedding()` | `tasks/embedding/embedding.py` | BAAI/bge-small-en-v1.5 |

---

## document-extraction

Extracts text content from uploaded documents, normalizing all formats to clean HTML.

**Input:**

```json
{
  "hash": "a1b2c3d4e5f6...",
  "extension": ".pdf"
}
```

**Output:**

```json
{
  "content": "<p>Extracted text...</p>",
  "pages": 12,
  "title": "Document Title",
  "author": "Author Name",
  "publication_date": "2024-01-15"
}
```

- `pages` is included for PDF and DOC/DOCX files.
- `title`, `author`, and `publication_date` are included for HTML files (extracted from meta tags).
- Supported extensions: `.pdf`, `.doc`, `.docx`, `.html`, `.htm`, `.txt`

See [Document Extraction](./document-extraction.md) for per-format details.

---

## detect-language

Detects the language of one or more text samples.

**Input:**

```json
{
  "samples": ["This is English text.", "Este es texto en español."]
}
```

**Output:**

```json
{
  "results": [
    { "text": "This is English text.", "language": "en" },
    { "text": "Este es texto en español.", "language": "es" }
  ]
}
```

Uses the `langdetect` library. Returns ISO 639-1 language codes.

---

## summarize

Generates a summary of the provided text, with cross-lingual support.

**Input:**

```json
{
  "content": "<p>Long document text...</p>",
  "sourceLanguage": "en",
  "targetLanguage": "es"
}
```

**Output:**

```json
{
  "response": "Resumen del documento..."
}
```

- Uses `facebook/mbart-large-50-one-to-many-mmt`.
- HTML tags are stripped before processing.
- Input is truncated to 1024 tokens.
- Summary length: 30-200 tokens, with beam search (4 beams) and no-repeat n-gram (size 3).

---

## translate

Translates a list of texts between language pairs.

**Input:**

```json
{
  "texts": [
    { "text": "Hello world", "path": "/optional/path" },
    "Plain string also accepted"
  ],
  "sourceLanguage": "en",
  "targetLanguage": "es"
}
```

**Output:**

```json
{
  "response": [
    {
      "translation_text": "Hola mundo",
      "original_text": "Hello world",
      "path": "/optional/path"
    },
    {
      "translation_text": "Cadena simple también aceptada",
      "original_text": "Plain string also accepted",
      "path": null
    }
  ]
}
```

- Uses `Helsinki-NLP/opus-mt-{source}-{target}` models (one model per language pair).
- Texts are processed in batches of 32.
- Supports both `targetLanguage` (string) and `targetLanguages` (array, uses the first element).
- The `path` field is passed through for client-side reference.

---

## entity-extraction

Extracts named entities from text using transformer-based NER.

**Input:**

```json
{
  "texts": [
    { "text": "John Doe works at Acme Corp in Madrid." },
    "Plain string also accepted"
  ]
}
```

**Output:**

```json
{
  "entities": [
    { "word": "John Doe", "entity": "PERSON" },
    { "word": "Acme Corp", "entity": "ORG" },
    { "word": "Madrid", "entity": "GPE" }
  ]
}
```

- Uses spaCy `en_core_web_trf` (transformer-based, loaded globally).
- Texts are processed in batches of 32 via `nlp.pipe()`.
- Filters out numerical/temporal entity types: CARDINAL, DATE, MONEY, ORDINAL, PERCENT, QUANTITY, TIME.
- Entities shorter than 2 characters are excluded.
- Duplicates are removed while preserving order.

---

## ingest-content

Ingests HTML content into the Qdrant vector database for RAG. Always deletes existing vectors for the source before upserting, keeping data in sync on re-ingestion.

**Input:**

```json
{
  "content": "<p>Document HTML content...</p>",
  "sourceType": "resource",
  "resourceId": 42,
  "projectId": "uuid-of-project"
}
```

The `sourceType` field controls how `source_id` is built:

| `sourceType` | Required field | `source_id` stored |
|---|---|---|
| `resource` (default) | `resourceId` | `"42"` |
| `doc` | `docId` | `"doc_42"` |
| `knowledge` | `knowledgeEntryId` | `"knowledge_42"` |

**Output:**

```json
{
  "success": true
}
```

- Cleans HTML and extracts text from block elements.
- Splits text into semantic chunks (target: `RAG_CHUNK_TARGET_WORDS` words, max: `RAG_CHUNK_MAX_WORDS`, overlap: `RAG_CHUNK_OVERLAP_WORDS`).
- Encodes each chunk with BAAI/bge-small-en-v1.5 (L2-normalized).
- Stores embeddings in Qdrant with metadata: `text`, `source_id`, `source_type`, `project_id`, `part_number`, `total_chunks`.

See [RAG Pipeline](./rag-pipeline.md) for details.

---

## delete-vectors

Deletes all Qdrant vectors belonging to a given source.

**Input:**

```json
{
  "sourceId": "42"
}
```

**Output:**

```json
{
  "success": true
}
```

- Removes all points from Qdrant where `source_id` matches the given value.
- Does not require any capability (any worker can handle it).

---

## search

Performs semantic search over ingested content.

**Input:**

```json
{
  "query": "climate change impact",
  "limit": 5,
  "projectId": "uuid-of-project",
  "score_threshold": 0.35
}
```

`projectId` filters results to a specific project. `score_threshold` overrides the global `RAG_SCORE_THRESHOLD` for this request.

**Output:**

```json
{
  "results": [
    {
      "text": "The impact of climate change on...",
      "score": 0.87,
      "metadata": {
        "text": "...",
        "source_id": "uuid",
        "project_id": "uuid",
        "part_number": 3
      }
    }
  ]
}
```

- Encodes the query with the same embedding model used for ingestion.
- Performs cosine similarity search in Qdrant.
- Returns up to `limit` results sorted by relevance score.

---

## ask

Answers questions using RAG (Retrieval-Augmented Generation).

**Input:**

```json
{
  "question": "What are the main findings of the report?",
  "projectId": "uuid-of-project"
}
```

`projectId` restricts retrieval to documents belonging to that project.

**Output:**

```json
{
  "response": "The main findings indicate that..."
}
```

- Encodes the question and retrieves top-k relevant chunks from Qdrant (default: 5).
- Deduplicates and re-ranks chunks by score.
- Builds a context prompt with the retrieved chunks.
- Generates an answer with the configured LLM (max `RAG_MAX_TOKENS` tokens, default: 1000).
- The prompt instructs the LLM to answer in the language the question is asked and to use only the provided context.

See [RAG Pipeline](./rag-pipeline.md) for the full flow.

---

## key-point

Extracts up to 5 key points from text content.

**Input:**

```json
{
  "content": "<p>Document text...</p>",
  "targetLanguage": "en"
}
```

**Output:**

```json
{
  "key_points": [
    "First key point sentence.",
    "Second key point sentence.",
    "Third key point sentence."
  ]
}
```

- HTML tags are stripped and entities are unescaped before processing.
- Uses the configured LLM to generate key points as complete sentences (3-10 words each).
- If the LLM is unavailable or produces insufficient results, falls back to heuristic extraction from the original text (splitting by sentence boundaries and filtering by word count).
- Returns up to 5 deduplicated key points.

---

## keywords

Extracts up to 10 keywords or short topic phrases from text content.

**Input:**

```json
{
  "content": "<p>Document text...</p>",
  "targetLanguage": "en"
}
```

**Output:**

```json
{
  "keywords": [
    "climate change",
    "renewable energy",
    "carbon emissions"
  ]
}
```

- HTML tags are stripped and entities are unescaped before processing.
- Uses the configured LLM (via chat completion or plain completion fallback) to generate comma-separated topics.
- Each keyword is truncated to a maximum of 3 words.
- If the LLM is unavailable (including when `llama-cpp-python` is not installed), falls back to heuristic extraction (first 3 words of each sentence).
- Returns up to 10 deduplicated keywords.

---

## embedding

Generates a vector embedding for a single text input.

**Input:**

```json
{
  "text": "Some text to encode"
}
```

**Output:**

```json
{
  "results": [0.023, -0.041, 0.087, "... (384 floats)"]
}
```

- Uses BAAI/bge-small-en-v1.5 (384-dimensional, L2-normalized).
- Returns the embedding as a list of floats.

---

## dataset-stats

Computes descriptive statistics for a dataset stored in the `datasets` / `dataset_records` tables.

**Input:**

```json
{
  "datasetId": 1,
  "filters": [
    { "field": "year", "operator": "gte", "value": 2020 }
  ]
}
```

`filters` is optional. Supported operators: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `contains`.

**Output:**

```json
{
  "total": 500,
  "filtered": 320,
  "fields": {
    "year": { "type": "number", "mean": 2022.1, "std": 1.4, "min": 2020, "max": 2025 },
    "country": { "type": "string", "unique": 42, "top": "Spain", "freq": 80 }
  }
}
```

- Reads schema and records directly from PostgreSQL.
- Builds a pandas DataFrame and computes per-field statistics (numeric: mean/std/min/max/percentiles; string: unique count and most-frequent value; boolean: true/false counts).
- Does not require any capability (any worker can handle it).
