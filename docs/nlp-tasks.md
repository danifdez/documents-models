# NLP Processing Tasks

This document covers the text analysis and transformation tasks that are not part of the [RAG pipeline](./rag-pipeline.md) or [document extraction](./document-extraction.md).

## Language Detection

**Execution type:** `detect-language`
**File:** `tasks/detect_language/detect_language.py`
**Library:** `langdetect`

Detects the language of one or more text samples and returns ISO 639-1 language codes (e.g., `en`, `es`, `fr`).

- Processes each sample in `payload["samples"]` independently.
- Returns a list of `{text, language}` objects.
- On error, returns `{"error": "..."}`.

## Summarization

**Step types:** `summarize-map`, `summarize-reduce`
**Files:** `tasks/summarize_map/summarize_map.py`, `tasks/summarize_reduce/summarize_reduce.py`
**Model:** local GGUF model configured independently for both step types

Backend creates the durable map/reduce graph. Models executes one bounded map
or one reduce assignment at a time and returns a canonical inference outcome.

**Processing steps:**

1. `summarize-map` summarizes exactly one chunk supplied by Backend.
2. Backend waits until every required map result is durably accepted.
3. Backend materializes the ordered response strings into the reduce payload.
4. `summarize-reduce` merges those partials into the final response.

**Notes:**

- Models does not chunk, enqueue children or finalize the execution.
- `config/tasks.json` has separate `summarize-map` and `summarize-reduce` entries.

## Translation

**Execution type:** `translate`
**File:** `tasks/translate/translate.py`
**Model:** `Helsinki-NLP/opus-mt-{source}-{target}` (prefix configured in `tasks.json`)

Translates a list of texts from a source language to a target language.

**Processing steps:**

1. Source and target language codes are extracted from the payload (with multiple fallback keys).
2. The appropriate OPUS model is loaded via `transformers.pipeline("translation", model=model_name)`.
3. Texts are normalized — both string items and `{text, path}` dict items are accepted.
4. Texts are completed in batches of 32.
5. Each result includes `translation_text`, `original_text`, and `path` (if provided).

**Notes:**

- Each language pair requires a separate model (e.g., `opus-mt-en-es`, `opus-mt-en-fr`).
- Models are downloaded from Hugging Face on first use.
- Supports `targetLanguage` (string) or `targetLanguages` (list, uses the first element).
- Returns `{"error": "..."}` if the model is not available for the requested language pair.

## Entity Extraction

**Step types:** `entity-extraction-map`, `entity-extraction-reduce`
**File:** `tasks/entities/entities.py`
**Model:** local Qwen LLM for map; deterministic Python for reduce

Backend creates the durable fan-out/fan-in graph. Models executes one bounded
map assignment or the deterministic reduce assignment; it does not coordinate
chunks or register a root handler.

**Processing steps:**

1. Backend splits extracted document text into chunks of at most 1500 words.
2. Each `entity-extraction-map` step receives one `content` string and returns
   an `entities` array.
3. Backend waits for all maps and materializes their arrays in dependency order.
4. `entity-extraction-reduce` validates the arrays and deduplicates names
   case-insensitively while preserving first appearance.

**Recognized entity types** (after filtering):

| Entity Type | Description |
|-------------|-------------|
| PERSON | People, including fictional |
| NORP | Nationalities, religious or political groups |
| FAC | Buildings, airports, highways, bridges, etc. |
| ORG | Companies, agencies, institutions, etc. |
| GPE | Countries, cities, states |
| LOC | Non-GPE locations (mountain ranges, bodies of water) |
| PRODUCT | Objects, vehicles, foods, etc. |
| EVENT | Named hurricanes, battles, wars, sports events, etc. |
| WORK_OF_ART | Titles of books, songs, etc. |
| LAW | Named documents made into laws |
| LANGUAGE | Any named language |

See [Entity Extraction](./tasks/entity-extraction.md) for the canonical map and
reduce payloads.

## Key Point Extraction

**Execution type:** `key-point`
**File:** `tasks/key_points/key_points.py`
**Model:** GGUF LLM configured in `config/tasks.json` (default: Qwen3-8B), with heuristic fallback

Extracts up to 5 concise key points from text content.

**Processing steps:**

1. HTML tags are stripped and HTML entities are unescaped.
2. A prompt is constructed asking the LLM for up to 5 key points (complete sentences, 3-10 words each) in the specified target language.
3. If the LLM is available:
   - Mistral-7B generates the response (max 1000 tokens).
   - Output is split into candidate lines/sentences.
4. If the LLM is unavailable or produces insufficient results:
   - Falls back to heuristic extraction: splits the original text by sentence-ending punctuation.
   - Filters sentences to those with 3-10 words.
5. Results are deduplicated and capped at 5 items.

## Keyword Extraction

**Step types:** `keywords-map`, `keywords-reduce`
**File:** `tasks/keywords/keywords.py`
**Model:** GGUF LLM for map; deterministic Python for reduce

Backend creates the durable fan-out/fan-in graph. Models executes one bounded
map assignment or the deterministic reduce assignment.

**Processing steps:**

1. Backend extracts and splits the document into chunks of at most 1500 words.
2. Each `keywords-map` step asks the LLM for candidates in the requested
   language and returns a `keywords` array.
3. Backend waits for every required map result and materializes the arrays in
   dependency order.
4. `keywords-reduce` counts each candidate once per chunk, ranks by frequency
   and first appearance, truncates phrases and applies the final item limit.

Models does not chunk, filter relevance or return a heuristic success when an
inference fails.

## Fallback Behavior Summary

| Task | LLM Available | LLM Unavailable |
|------|---------------|-----------------|
| `key-point` | Configured LLM generates key points, supplemented by heuristics if < 5 results | Pure heuristic: sentence splitting + word count filtering |
| `keywords-map` | Configured LLM generates keyword candidates | Step fails explicitly |
| `ask` | Configured LLM generates answer from context | Task fails (no fallback) |

## Dataset Statistics

**Execution type:** `dataset-stats`
**File:** `tasks/dataset_stats/stats.py`
**Library:** pandas, scipy

Computes descriptive statistics for a dataset and its records stored in the PostgreSQL `datasets` and `dataset_records` tables.

**Processing steps:**

1. Schema and records are fetched directly from PostgreSQL.
2. Records are assembled into a pandas DataFrame with type coercion based on the schema field types.
3. Optional filters are applied (operators: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `contains`).
4. Per-field statistics are computed:
   - **Numeric fields**: mean, std, min, max, percentiles (25th, 50th, 75th)
   - **String fields**: unique count, most-frequent value and its frequency
   - **Boolean fields**: true count, false count
5. Returns total record count, filtered record count, and per-field stats.
