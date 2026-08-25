## Keywords

Keyword extraction is a durable map/reduce workflow coordinated by Backend.
Models exposes `keywords-map` and `keywords-reduce`; it does not register a
`keywords` root handler.

### Map step

Each `keywords-map` assignment receives one bounded chunk and its requested
output language:

```json
{
  "content": "Machine learning uses labeled training data.",
  "targetLanguage": "en"
}
```

The chunk must be non-empty and contain no more than `max_input_words` words
(1500 by default). The configured LLM returns comma- or newline-separated
candidates, which the handler exposes as:

```json
{
  "keywords": ["machine learning", "labeled training data"]
}
```

An unavailable model, invalid input or empty generation fails the step. Models
does not hide these failures behind a heuristic result.

### Reduce step

Backend waits for every required map result and materializes the ordered
`keywords` arrays into a deterministic `keywords-reduce` assignment:

```json
{
  "partials": [
    ["durable workflows", "PostgreSQL"],
    ["postgresql", "execution evidence"]
  ]
}
```

The reduce step counts a candidate at most once per chunk, compares candidates
case-insensitively, ranks by frequency and then by first appearance, and
returns the first `max_items` values. Each value is limited to
`max_words_per_item` words; defaults are 10 values and 3 words.
