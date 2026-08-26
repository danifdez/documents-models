# Extract key points

Key-point extraction produces a short list of the document's main statements in the requested language.

## How it works

Documents divides long content into sections of at most 1,500 words by default. Candidate sentences are produced for each section, combined in document order, and deduplicated. A final refinement selects up to five points by default.

The result contains complete, bounded sentences rather than isolated keywords.

Invalid or empty input, an unavailable language model, or empty model output causes the action to fail explicitly. There is no heuristic or semantic-search fallback.
