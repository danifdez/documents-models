## Search

The **search** task performs a semantic search over the content indexed in the vector database. It finds the most relevant text snippets for a given query, optionally scoped to a specific project.

### What it does

The query text is converted into a vector and compared against all stored document chunks. The most relevant results are returned, each with a relevance score and metadata about where the text came from. Results are optionally re-ranked for improved quality.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | The search query in natural language |
| `limit` | number | Yes | Maximum number of results to return |
| `projectId` | number | No | Limits the search to a specific project's content |
| `score_threshold` | number | No | Minimum relevance score for a result to be included (0 to 1) |

### Returns

```json
{
  "results": [
    {
      "text": "The relevant text snippet...",
      "score": 0.87,
      "metadata": {
        "source_id": "42",
        "source_type": "resource",
        "project_id": "5",
        "part_number": 2,
        "total_chunks": 8
      }
    }
  ]
}
```

Results are ordered by relevance score (highest first).

### Example

**Input:**

```json
{
  "query": "sustainable energy policies in Europe",
  "limit": 3,
  "projectId": 12,
  "score_threshold": 0.4
}
```

**Output:**

```json
{
  "results": [
    {
      "text": "The European Green Deal aims to make Europe climate-neutral by 2050...",
      "score": 0.91,
      "metadata": {
        "source_id": "87",
        "source_type": "resource",
        "project_id": "12",
        "part_number": 1,
        "total_chunks": 5
      }
    }
  ]
}
```
