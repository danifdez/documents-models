## Key Points

The **key-point** task extracts the most important ideas from a piece of text as a short list of concise statements. It is useful for quickly understanding what a document is about without reading the full content.

### What it does

The task uses a language model to identify and return the key takeaways from the provided text. Each key point is a short, clear statement (typically between 3 and 10 words). If the language model is unavailable, the task falls back to extracting the first sentence fragment from the text.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The text to extract key points from |
| `targetLanguage` | string | No | Language code for the output (e.g. `en`, `es`). Defaults to `en` |

### Returns

```json
{
  "key_points": [
    "First important point",
    "Second important point",
    "Third important point"
  ]
}
```

On error:

```json
{
  "error": "Error extracting key points: <reason>"
}
```

### Example

**Input:**

```json
{
  "content": "Remote work has become increasingly common since 2020. Studies show that employees working from home report higher productivity and better work-life balance. However, companies face challenges in maintaining team cohesion and company culture. Hybrid models, combining remote and in-office days, are emerging as a popular compromise.",
  "targetLanguage": "en"
}
```

**Output:**

```json
{
  "key_points": [
    "Remote work has grown since 2020",
    "Higher productivity reported by remote workers",
    "Challenges in team cohesion",
    "Hybrid models gaining popularity"
  ]
}
```
