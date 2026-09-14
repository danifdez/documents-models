# Extract relationships

Relationship extraction proposes typed links between named entities already found in a document section. It uses the multilingual `joeddav/xlm-roberta-large-xnli` sequence classifier at the pinned revision configured in `config/tasks.json`; it does not use the generative LLM.

## Relation catalogue

The worker only returns relations from the configured catalogue. The default catalogue contains 20 general predicates covering employment and membership, leadership and ownership, corporate structure and transactions, partnerships and competition, locations and containment, creation and use, family links, education, life events, and government. Authorship, development, and production are intentionally represented by the broader `created` predicate because the general-purpose classifier does not separate those near-synonyms reliably.

Each catalogue entry maps a natural-language hypothesis to a stable snake-case predicate. Built-in predicates also constrain valid source and target entity types. Symmetric entries, such as `partnered_with`, are deduplicated regardless of entity order. The catalogue, score thresholds, pair limit, and maximum results per pair can be changed under `relationship-extraction-map` in `config/tasks.json`.

## Language coverage

The classifier was trained for natural-language inference in these 15 XNLI languages:

- Arabic, Bulgarian, Chinese, English, French, German, Greek, Hindi;
- Russian, Spanish, Swahili, Thai, Turkish, Urdu, and Vietnamese.

These 15 languages have model-level XNLI support. Documents has smoke-tested the calibrated relationship task in English, Spanish, and French; the other 12 still require task-specific evaluation before they should be described as validated by Documents. XLM-R was pretrained on 100 languages, so further languages may work, but they remain best effort. Relation labels remain internal English hypotheses; the classifier supports premises and hypotheses written in different languages.

## Processing and result

Only entities whose names occur in the current bounded section are paired. By default the worker evaluates at most 32 unordered pairs and accepts one directed relation per pair. A candidate must have an entailment score of at least `0.8` and improve its log-odds by at least `2.5` over a neutral sentence that only mentions the two entities. This conservative margin favors precision over recall because accepted relationships can become persistent graph data. When semantically overlapping candidates are within `0.7` log-odds, the more specific predicate wins. Accepted results retain the existing shape:

```json
{
  "subject": "Ada Lovelace",
  "predicate": "created",
  "object": "Analytical Engine",
  "confidence": 0.94,
  "context": "Ada Lovelace created the Analytical Engine."
}
```

Map results are combined by the replayable reduce step, which preserves input order and retains the highest-confidence duplicate. The backend remains responsible for validating and persisting the final effects.

## Limitations

This first version is closed-domain classification, not open relation discovery. It cannot emit a relationship that is absent from the configured catalogue. The type constraints assume the entity categories produced by Documents; custom categories do not receive that filter. XNLI measures textual entailment rather than relationship extraction specifically, so the calibration and catalogue require evaluation against representative Documents data before the feature should be treated as high-confidence automation.
