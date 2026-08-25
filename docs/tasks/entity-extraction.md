## Entity Extraction

Entity extraction is a durable map/reduce workflow coordinated by Backend.
Models exposes two self-contained step handlers; it does not expose an
`entity-extraction` root handler.

### Map step

`entity-extraction-map` analyzes one bounded document chunk with the local
Qwen LLM.

```json
{
  "content": "Marie Curie worked at the Sorbonne."
}
```

The chunk must be non-empty and contain no more than `max_input_words` words
(1500 by default). The result is:

```json
{
  "entities": [
    { "word": "Marie Curie", "entity": "PERSON" },
    { "word": "Sorbonne", "entity": "ORG" }
  ]
}
```

### Reduce step

After every required map result is durably accepted, Backend materializes the
ordered `entities` arrays into an `entity-extraction-reduce` code step:

```json
{
  "partials": [
    [{ "word": "Marie Curie", "entity": "PERSON" }],
    [
      { "word": "marie curie", "entity": "PERSON" },
      { "word": "Sorbonne", "entity": "ORG" }
    ]
  ]
}
```

The reduce step validates every entity, concatenates partials in dependency
order and deduplicates names case-insensitively while preserving their first
appearance. It returns the same `entities` result shape as a map step.

Recognized labels are `PERSON`, `ORG`, `GPE`, `LOC`, `NORP`, `EVENT`, `FAC`,
`PRODUCT`, `WORK_OF_ART`, `LANGUAGE` and `LAW`. `PERSON`, `ORG` and the other
proper-name labels require at least one uppercase character; `NORP` and
`LANGUAGE` are exempt so lowercase multilingual forms remain valid.
