# RAG pipeline

RAG is split at the Backend–Models authority boundary. Backend owns vector
storage and scope; Models owns embedding, ranking and answer generation.

```text
ingest-content assignment
  -> Models cleans, chunks and embeds
  -> Models uploads deterministic vector_points artifacts
  -> StepResult references the artifacts and returns their point count
  -> Backend atomically replaces the source rows in pgvector

search/ask request
  -> Backend reads scope-checked vector candidates
  -> Backend freezes them in a vector_candidates artifact
  -> Models embeds the query and ranks only that snapshot
  -> search returns ranked chunks
  -> ask builds grounded context and performs one answer inference
```

Models never connects to PostgreSQL or Apache AGE. For `ask`, Backend also
resolves any mentioned graph entities and includes the project-scoped
relationships in `graphContext` before granting the step.

## Ingestion

`ingest-content` cleans HTML with `services/text.py`, splits it using the RAG
chunk limits, and embeds every chunk with
`intfloat/multilingual-e5-small`. Its output is:

```json
{
  "sourceId": "resource_42",
  "chunks": 1,
  "pointCount": 1
}
```

The `StepResult` references ordered `vector_points` artifacts containing:

```json
{
  "points": [
    {
      "id": "resource_42:1",
      "embedding": [0.1],
      "payload": {
        "text": "...",
        "source_id": "resource_42",
        "source_type": "resource",
        "part_number": 1,
        "total_chunks": 1
      }
    }
  ]
}
```

The real embedding has exactly 384 finite values. Point identities are stable
for a source and chunk position. Models does not delete or upsert rows.

## Retrieval

`search`, `ask` and `indexed-file-search` require a bounded
candidate snapshot. In production it arrives as the `vector_candidates`
artifact; direct harness cases may declare the same `candidates` array in the
payload.

Models validates every 384-dimensional vector, calculates cosine similarity,
applies the requested threshold, and uses a deterministic score/id ordering.
It cannot observe candidates outside the Backend-selected scope.

`ask` then deduplicates ranked chunks, appends the supplied `graphContext`,
builds the prompt and performs the configured LLM inference.

## Configuration

The `rag` block controls chunk sizes, default result limit, response budget and
score threshold. Database hosts, vector table names and graph settings are not
Models configuration; they belong exclusively to Backend.
