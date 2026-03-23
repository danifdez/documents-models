# NLP Processing Tasks

This document covers the text analysis and transformation tasks that are not part of the [RAG pipeline](./rag-pipeline.md) or [document extraction](./document-extraction.md).

## Language Detection

**Job type:** `detect-language`
**File:** `tasks/detect_language/detect_language.py`
**Library:** `langdetect`

Detects the language of one or more text samples and returns ISO 639-1 language codes (e.g., `en`, `es`, `fr`).

- Processes each sample in `payload["samples"]` independently.
- Returns a list of `{text, language}` objects.
- On error, returns `{"error": "..."}`.

## Summarization

**Job type:** `summarize`
**File:** `tasks/summarize/summarize.py`
**Model:** configured in `config/models.json` (default: `facebook/mbart-large-50-one-to-many-mmt`)

Generates a summary of the input text with cross-lingual support (source and target language can differ).

**Processing steps:**

1. HTML tags are stripped from the content using regex.
2. The tokenizer is configured with the source language (`sourceLanguage` + `_XX` suffix).
3. Input is tokenized with a maximum length of 1024 tokens (truncated if longer).
4. The model generates a summary with:
   - Beam search: 4 beams
   - Length: 30-200 tokens
   - No-repeat n-gram: size 3
   - Forced BOS token for the target language
5. The summary is decoded and returned.

**Notes:**

- The model and tokenizer are loaded on first use (lazy loading).
- mBART-50 supports 50 languages. Language codes use the `xx_XX` format internally (e.g., `en_XX`, `es_XX`).

## Translation

**Job type:** `translate`
**File:** `tasks/translate/translate.py`
**Model:** `Helsinki-NLP/opus-mt-{source}-{target}` (prefix configured in `models.json`)

Translates a list of texts from a source language to a target language.

**Processing steps:**

1. Source and target language codes are extracted from the payload (with multiple fallback keys).
2. The appropriate OPUS model is loaded via `transformers.pipeline("translation", model=model_name)`.
3. Texts are normalized — both string items and `{text, path}` dict items are accepted.
4. Texts are processed in batches of 32.
5. Each result includes `translation_text`, `original_text`, and `path` (if provided).

**Notes:**

- Each language pair requires a separate model (e.g., `opus-mt-en-es`, `opus-mt-en-fr`).
- Models are downloaded from Hugging Face on first use.
- Supports `targetLanguage` (string) or `targetLanguages` (list, uses the first element).
- Returns `{"error": "..."}` if the model is not available for the requested language pair.

## Entity Extraction

**Job type:** `entity-extraction`
**File:** `tasks/entities/entities.py`
**Model:** spaCy model configured in `config/models.json` (default: `en_core_web_sm`; set to `en_core_web_trf` for transformer-based NER)

Extracts named entities (persons, organizations, locations, etc.) from text.

**Processing steps:**

1. The spaCy model is loaded globally at module import (not per-request).
2. Input texts are normalized — both strings and `{text}` dicts are accepted.
3. Texts are processed in batches of 32 via `nlp.pipe()`.
4. Entities are filtered:
   - Excluded types: CARDINAL, DATE, MONEY, ORDINAL, PERCENT, QUANTITY, TIME
   - Minimum length: 2 characters
5. Duplicates are removed while preserving discovery order.

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

## Key Point Extraction

**Job type:** `key-point`
**File:** `tasks/key_points/key_points.py`
**Model:** GGUF LLM configured in `config/models.json` (default: Phi-4-mini-instruct), with heuristic fallback

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

**Job type:** `keywords`
**File:** `tasks/keywords/keywords.py`
**Model:** GGUF LLM configured in `config/models.json` (default: Phi-4-mini-instruct), with heuristic fallback

Extracts up to 10 keywords or short topic phrases from text content.

**Processing steps:**

1. HTML tags are stripped and HTML entities are unescaped.
2. A prompt is constructed asking the LLM for up to 10 topics (1-3 words each), comma-separated, in the same language as the input.
3. If the LLM is available:
   - Tries chat completion first, then falls back to plain completion.
   - Output is split by commas/newlines and cleaned of bullet markers.
4. If the LLM is unavailable (including when `llama-cpp-python` is not installed):
   - Falls back to heuristic extraction: takes the first 3 words of each sentence.
5. Each keyword is truncated to a maximum of 3 words.
6. Results are deduplicated and capped at 10 items.

## Fallback Behavior Summary

| Task | LLM Available | LLM Unavailable |
|------|---------------|-----------------|
| `key-point` | Configured LLM generates key points, supplemented by heuristics if < 5 results | Pure heuristic: sentence splitting + word count filtering |
| `keywords` | Configured LLM generates keywords via chat or completion API | Pure heuristic: first 3 words of each sentence |
| `ask` | Configured LLM generates answer from context | Task fails (no fallback) |

## Dataset Statistics

**Job type:** `dataset-stats`
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
