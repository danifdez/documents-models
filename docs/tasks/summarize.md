## Summarize

The **summarize** task generates a condensed summary of a document or piece of text. It supports cross-lingual summarization, meaning the input and output can be in different languages.

### What it does

The task strips any HTML formatting from the input, then uses a multilingual sequence-to-sequence model to produce a shorter version that captures the main ideas of the original content.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The text or HTML content to summarize |
| `sourceLanguage` | string | Yes | Language code of the input text (e.g. `en`, `es`, `fr`) |
| `targetLanguage` | string | Yes | Language code for the output summary (e.g. `en`, `es`) |

Language codes follow the ISO 639-1 standard.

### Returns

```json
{
  "response": "The generated summary text."
}
```

### Example

**Input:**

```json
{
  "content": "<p>Artificial intelligence has transformed the way we interact with technology. From voice assistants to recommendation systems, AI is now embedded in everyday life. Researchers continue to push boundaries, exploring new architectures and training methods that improve model accuracy and efficiency. Despite significant progress, challenges around interpretability, bias, and energy consumption remain open problems in the field.</p>",
  "sourceLanguage": "en",
  "targetLanguage": "en"
}
```

**Output:**

```json
{
  "response": "AI has transformed technology and everyday life, but challenges like bias and energy use remain."
}
```
