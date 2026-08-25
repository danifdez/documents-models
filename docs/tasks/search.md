# Search

The `search` step embeds a query and ranks the bounded candidate snapshot that
Backend attached to the assignment. It never reads pgvector or widens the
authorized project scope.

## Inputs

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Natural-language query |
| `limit` | integer | yes | Maximum result count |
| `score_threshold` | number | no | Minimum cosine score |
| `vector_candidates` | artifact | yes | JSON object containing at most 5,000 384-dimensional candidates |

Backend resolves `projectId` while building the candidate artifact; Models does
not use it as an authorization control.

## Result

```json
{
  "results": [
    {
      "text": "The relevant text snippet...",
      "score": 0.87,
      "metadata": {
        "source_id": "resource_42",
        "source_type": "resource",
        "project_id": 5,
        "part_number": 2,
        "total_chunks": 8
      }
    }
  ]
}
```

Results are sorted deterministically by descending cosine similarity and then
candidate ID. The assignment fails if its artifact is missing, malformed,
oversized, or contains embeddings with a dimension other than 384.
