# Available processing capabilities

The capabilities below are available when the installation has enabled them and a compatible processor is online.

## Documents and language

| Capability | What users receive |
|---|---|
| [Document extraction](./document-extraction.md) | Clean readable content and available metadata from supported files. |
| [Language detection](./tasks/detect-language.md) | A two-letter language code for each text sample. |
| [Summarization](./tasks/summarize.md) | One concise summary in the requested language. |
| [Translation](./tasks/translate.md) | Translated text in the same order as the originals. |
| [Entity extraction](./tasks/entity-extraction.md) | People, organizations, places, and other recognized names for review. |
| [Key points](./tasks/key-point.md) | A short refined list of the document's main points. |
| [Keywords](./tasks/keywords.md) | A ranked list of concise recurring concepts. |
| Date extraction | Normalized dates, ranges, precision, location in the text, and unresolved relative expressions. |
| [Transcription](./tasks/transcribe.md) | Speech converted to text with detected language, confidence, and duration. |

Relative dates such as “next Tuesday” require an anchor date. Without one, the expression remains visible but unresolved. Date extraction accepts valid text with no dates and returns an empty result.

## Semantic search and answers

| Capability | What users receive |
|---|---|
| [Content indexing](./tasks/ingest-content.md) | Project content prepared for meaning-based retrieval. |
| [Semantic search](./tasks/search.md) | Relevant passages ranked by similarity. |
| [Ask](./tasks/ask.md) | A natural-language answer grounded in supplied project context. |
| [Embedding](./tasks/embedding.md) | An internal semantic representation used for comparison and retrieval. |

Search and answers use only candidates selected from the current project. They do not query unrelated projects or external sources.

## Dataset analysis

| Capability | What users receive |
|---|---|
| [Summary](./tasks/summary.md) | Descriptive statistics for every field. |
| [Query](./tasks/query.md) | Filtered, selected, grouped, or aggregated records. |
| [Chart](./tasks/chart.md) | Data prepared for bar, line, pie, or scatter charts. |
| [Distribution](./tasks/distribution.md) | A histogram or ranked category frequencies. |
| [Group by](./tasks/group-by.md) | A numeric aggregation for each category. |
| [Correlation](./tasks/correlation.md) | Pearson correlation, significance, and a regression line for two fields. |
| [Correlation matrix](./tasks/correlation-matrix.md) | Pairwise correlations across numeric fields. |
| [Pivot table](./tasks/pivot-table.md) | A cross-tabulation with row, column, and grand totals. |
| [Time series](./tasks/time-series.md) | Period averages and an overall trend. |
| [Outliers](./tasks/outliers.md) | Unusual numeric values identified using the interquartile range. |

Dataset analysis operates on the project-scoped snapshot supplied by Documents and never opens the application data store directly.
