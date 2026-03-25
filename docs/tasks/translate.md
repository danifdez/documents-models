## Translate

The **translate** task translates a list of texts from one language to another. It supports plain strings as well as structured objects that carry additional metadata like file paths.

### What it does

Each text in the input list is translated from the source language to the target language using a neural machine translation model. The original text and any associated metadata are preserved in the output alongside the translation.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `texts` | array | Yes | List of texts to translate. Each item can be a plain string or an object with a `text` field |
| `sourceLanguage` | string | No | Language code of the source text (e.g. `en`). Defaults to `en` |
| `targetLanguage` | string | No | Language code for the translation (e.g. `es`). Defaults to `es` |
| `targetLanguages` | array | No | List of target language codes. Only the first is used |

Language codes follow the ISO 639-1 standard.

### Returns

```json
{
  "response": [
    {
      "translation_text": "Translated text here",
      "original_text": "Original text here",
      "path": null
    }
  ]
}
```

Each item in `response` corresponds to an item in the input list, preserving the same order. The `path` field is only present when the input item included it.

On error:

```json
{
  "error": "Error during translation: <reason>"
}
```

### Example

**Input:**

```json
{
  "texts": [
    { "text": "The report is ready for review.", "path": "/docs/report.pdf" },
    "Please submit your feedback."
  ],
  "sourceLanguage": "en",
  "targetLanguage": "es"
}
```

**Output:**

```json
{
  "response": [
    {
      "translation_text": "El informe está listo para su revisión.",
      "original_text": "The report is ready for review.",
      "path": "/docs/report.pdf"
    },
    {
      "translation_text": "Por favor, envíe sus comentarios.",
      "original_text": "Please submit your feedback.",
      "path": null
    }
  ]
}
```
