## Summarize

Documents materializes summarization as a durable workflow. Models exposes two
self-contained capabilities: **summarize-map** summarizes one bounded chunk and
**summarize-reduce** merges the ordered partials.

### What it does

Backend owns HTML-to-text conversion, chunking, step identities and dependencies.
Models uses the configured local GGUF model for map and reduce inference and
never creates child executions.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Map only | One bounded plain-text chunk |
| `partials` | string[] | Reduce only | Ordered map responses materialized by Backend |
| `sourceLanguage` | string | No | Language code of the input text |
| `targetLanguage` | string | Yes | Language code for the output summary |

Language codes follow the ISO 639-1 standard.

### Returns

```json
{
  "response": "The generated summary text."
}
```

### Example

**Map input:**

```json
{
  "content": "Artificial intelligence has transformed everyday technology.",
  "sourceLanguage": "en",
  "targetLanguage": "en"
}
```

**Reduce input:**

```json
{
  "partials": ["First partial.", "Second partial."],
  "targetLanguage": "en"
}
```

**Output:**

```json
{
  "response": "AI has transformed technology and everyday life, but challenges like bias and energy use remain."
}
```
