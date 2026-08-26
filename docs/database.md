# Data and privacy

The AI processing service does not have direct access to the Documents database, semantic index, or relationship graph.

## What processing receives

For each action, Documents supplies only the information needed to calculate that result, such as:

- a bounded section of text;
- an uploaded file for extraction or transcription;
- a project-scoped dataset snapshot;
- a limited set of search candidates already selected from the current project;
- relevant project relationships for a question.

Large inputs are transferred specifically for the active processing attempt. Missing or malformed required material causes the action to fail instead of falling back to another data source.

## What processing returns

The service returns calculated values such as extracted text, summaries, translations, entities, rankings, embeddings, transcripts, or statistics. Documents validates those results and is solely responsible for saving changes to project data.

## Search scope

Semantic search and question answering cannot widen the supplied candidate set. The processing service ranks only the project-scoped candidates provided by Documents, so it cannot discover content from another project through direct data access.
