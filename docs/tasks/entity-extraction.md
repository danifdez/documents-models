## Entity Extraction

The **entity-extraction** task identifies and extracts named entities from text, such as people, organizations, locations, and more. It processes multiple texts at once and returns all unique entities found.

### What it does

Each text in the input list is analyzed to detect named entities — meaningful proper nouns that can be classified into categories. The task uses a transformer-based NLP model for high accuracy and processes texts in batches for performance. Common numerical and temporal entity types (dates, quantities, money, etc.) are filtered out by default.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `texts` | array | Yes | List of texts to analyze. Each item can be a plain string or an object with a `text` field |

### Returns

```json
{
  "entities": [
    { "word": "Entity name", "entity": "ENTITY_TYPE" }
  ]
}
```

**Common entity types:**

| Type | Description |
|------|-------------|
| `PERSON` | People's names |
| `ORG` | Organizations, companies |
| `GPE` | Countries, cities, states |
| `LOC` | Non-GPE locations (rivers, mountains) |
| `EVENT` | Named events |
| `PRODUCT` | Products and objects |
| `WORK_OF_ART` | Titles of books, films, etc. |
| `LAW` | Named legal documents |

Duplicate entities (same word and type) are removed while preserving the order of first appearance.

### Example

**Input:**

```json
{
  "texts": [
    { "text": "Marie Curie was born in Warsaw and later worked at the University of Paris." },
    "Apple Inc. was founded by Steve Jobs in California."
  ]
}
```

**Output:**

```json
{
  "entities": [
    { "word": "Marie Curie", "entity": "PERSON" },
    { "word": "Warsaw", "entity": "GPE" },
    { "word": "University of Paris", "entity": "ORG" },
    { "word": "Apple Inc.", "entity": "ORG" },
    { "word": "Steve Jobs", "entity": "PERSON" },
    { "word": "California", "entity": "GPE" }
  ]
}
```
