## Keywords

The **keywords** task extracts the most representative terms or short phrases from a piece of text. It can return keywords in a specified language and is useful for tagging, indexing, and content classification.

### What it does

The task sends the text to a language model with instructions to identify the most significant keywords. Results are cleaned and deduplicated, keeping only concise terms (up to 3 words each). If the language model is unavailable, the task falls back to a heuristic approach that extracts the first words of each sentence.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The text to extract keywords from |
| `targetLanguage` | string | No | Language code for the output keywords (e.g. `en`, `es`). Defaults to `auto` |

### Returns

```json
{
  "keywords": ["keyword one", "keyword two", "keyword three"]
}
```

On error:

```json
{
  "error": "Error extracting keywords: <reason>"
}
```

### Example

**Input:**

```json
{
  "content": "Machine learning models require large amounts of labeled training data. Data labeling is a time-consuming process often done by human annotators. Active learning techniques can reduce the labeling effort by selecting the most informative samples.",
  "targetLanguage": "en"
}
```

**Output:**

```json
{
  "keywords": ["machine learning", "labeled training data", "data labeling", "active learning", "human annotators"]
}
```
