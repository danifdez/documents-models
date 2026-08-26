# Semantic search

Semantic search finds project passages related to the meaning of a natural-language query, even when they do not use exactly the same words.

## Scope

Documents selects a bounded set of candidates from the current project before processing begins. The processing service ranks only those candidates and cannot widen the search to another project, the application database, or the web.

## Result

Each result includes:

- the matching text passage;
- a relevance score;
- source information and its position within the indexed content.

Results are ordered by descending similarity. Equal scores use a stable ordering, so the same query over the same candidates gives a consistent result order.

The default limit is five results and the default minimum relevance score is 0.35. Malformed or oversized candidate data causes the action to fail rather than returning partial rankings.
