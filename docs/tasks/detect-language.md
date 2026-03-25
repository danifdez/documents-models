## Detect Language

The **detect-language** task identifies the language of one or more text samples. It returns a language code for each input, making it easy to route content to the appropriate language-specific processing pipeline.

### What it does

Each text sample is analyzed and assigned an ISO 639-1 language code (e.g. `en` for English, `es` for Spanish, `fr` for French). Multiple samples can be sent in a single request.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `samples` | array of strings | Yes | List of text samples to detect the language of |

### Returns

```json
{
  "results": [
    { "text": "The original text sample", "language": "en" }
  ]
}
```

Each item in `results` contains the original text and its detected language code. Results are returned in the same order as the input samples.

On error:

```json
{
  "error": "Error detecting language: <reason>"
}
```

### Example

**Input:**

```json
{
  "samples": [
    "The quick brown fox jumps over the lazy dog.",
    "La inteligencia artificial está cambiando el mundo.",
    "Die Sonne scheint heute sehr hell."
  ]
}
```

**Output:**

```json
{
  "results": [
    { "text": "The quick brown fox jumps over the lazy dog.", "language": "en" },
    { "text": "La inteligencia artificial está cambiando el mundo.", "language": "es" },
    { "text": "Die Sonne scheint heute sehr hell.", "language": "de" }
  ]
}
```
