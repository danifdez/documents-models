## Key Points

Key-point extraction is a durable inference map/reduce workflow coordinated by
Backend. Models exposes `key-point-map` and `key-point-reduce`; it does not
register a `key-point` root handler.

### Map step

Each `key-point-map` assignment receives one bounded document chunk:

```json
{
  "content": "Backend persists workflow state before dispatching work.",
  "targetLanguage": "en"
}
```

The chunk must be non-empty and contain no more than `max_input_words` words
(1500 by default). The handler performs one LLM inference and returns unique
candidate sentences within the configured word bounds:

```json
{
  "key_points": ["Backend persists workflow state before dispatch"]
}
```

### Reduce step

Backend waits for every required map result and materializes their ordered
arrays into one `key-point-reduce` assignment:

```json
{
  "targetLanguage": "en",
  "partials": [
    ["Backend persists workflow state before dispatch"],
    ["Workers execute bounded assignments with leases"]
  ]
}
```

The reduce handler removes exact duplicates, validates sentence lengths and
performs one LLM inference with `refine_prompt.md` to select at most
`max_items` final points. Both handlers fail explicitly on invalid input,
unavailable inference or empty output; there is no heuristic or embeddings
fallback.
