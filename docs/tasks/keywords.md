# Extract keywords

Keyword extraction produces a ranked list of concise concepts in the requested language.

## How it works

Documents divides long content into sections of at most 1,500 words by default. Candidate phrases are extracted from each section, then compared without regard to capitalization.

A candidate is counted at most once per section. Final ranking uses frequency across sections and then first appearance in the document.

By default, the result includes up to ten keywords or phrases, each limited to three words.

Invalid input, an unavailable language model, or empty generation causes the action to fail explicitly rather than returning heuristic keywords.
