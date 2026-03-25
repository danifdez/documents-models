## Embedding

The **embedding** task converts a piece of text into a numerical vector representation. These vectors capture the semantic meaning of the text and are used internally for similarity search, document indexing, and retrieval operations.

### What it does

Given a text string, this task returns a fixed-size vector of floating point numbers. Texts with similar meaning produce vectors that are close to each other in vector space, which enables semantic search and comparison.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | The text to convert into an embedding vector |

### Returns

```json
{
  "results": [0.021, -0.045, 0.113, ...]
}
```

The `results` field contains a list of floating point numbers representing the embedding vector. The vector length depends on the configured embedding model (typically 384 dimensions for `BAAI/bge-small-en-v1.5`).

### Example

**Input:**

```json
{
  "text": "Climate change and its effects on global agriculture"
}
```

**Output:**

```json
{
  "results": [0.0213, -0.0451, 0.1132, 0.0874, -0.0329, "..."]
}
```
