# Semantic search and grounded answers

Documents uses semantic retrieval to find passages by meaning rather than only by exact words. This supports both project search and question answering.

## Preparing content

When content is indexed, Documents:

1. removes markup that does not contribute useful text;
2. divides the content into overlapping passages;
3. creates a semantic representation for each passage;
4. stores the approved index entries with their source and position.

The default passage target is 150 words, with a maximum of 250 words and a 30-word overlap. Empty content produces no index entries.

## Searching

For a search, Documents first selects a bounded set of candidates from the current project. The processing service compares the query with only that snapshot, applies the relevance threshold, and orders matches by score. Ties are resolved consistently so repeated searches over the same snapshot remain stable.

The default search returns up to five passages with a minimum relevance score of 0.35. A request can use a different result limit or threshold when the feature allows it.

## Asking a question

Question answering uses the best-ranked passages as context. Duplicate passages are removed. Documents can also supply relationships between entities that were resolved inside the same project.

The language model is instructed to answer in the language of the question and to rely on the supplied context. If no relevant content is found, Documents says that it could not find enough information rather than inventing project evidence.

## Scope and limitations

- Only content selected from the current project is considered.
- The processing service cannot search the database, relationship graph, another project, or the web.
- Semantic similarity identifies related meaning but does not prove that a passage is factually correct.
- An answer is generated from retrieved context and should be checked against its source material for important decisions.
