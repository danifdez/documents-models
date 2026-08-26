# Prepare content for semantic search

Content indexing makes a resource, editable document, or knowledge entry available to meaning-based search and grounded question answering.

## What happens

Documents cleans the supplied text, divides it into overlapping passages, and creates a semantic representation for each passage. Every passage keeps its source, position, and total passage count.

The default passage target is 150 words, with a maximum of 250 words and a 30-word overlap.

## Result

A successful action reports the indexed source and how many passages were prepared. Empty content produces zero passages and does not create search entries.

Documents validates and saves the new entries as one replacement of the source's previous semantic index. The processing service does not open or modify the search database itself.
