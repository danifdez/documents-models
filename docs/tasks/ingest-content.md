# Ingest Content

`ingest-content` is a self-contained embedding step. It cleans and chunks the
supplied content, calculates 384-dimensional embeddings and returns vector
points. Backend owns replacement and persistence in pgvector.

## Input

| Parameter | Type | Required | Description |
|---|---|---|---|
| `content` | string | yes | HTML or text to index |
| `projectId` | number | no | Scope recorded in point metadata |
| `sourceType` | string | no | `resource`, `doc` or `knowledge`; defaults to `resource` |
| `resourceId` | number | conditional | Required for a resource |
| `docId` | number | conditional | Required for a doc |
| `knowledgeEntryId` | number | conditional | Required for a knowledge entry |

## Output

```json
{
  "sourceId": "resource_42",
  "chunks": 1,
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

The abbreviated vector above represents 384 finite values. Empty content
returns zero points. Cleanup is not a Models task: Backend deletes or replaces
vectors directly as owner of the domain store.
