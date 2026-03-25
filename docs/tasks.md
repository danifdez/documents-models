## Tasks

This section documents all available tasks in the models service. Each task is a job handler that receives a payload and returns a result.

Tasks are grouped into two categories: **NLP tasks** for processing natural language content, and **Dataset tasks** for analyzing structured data.

---

### NLP Tasks

These tasks work with unstructured text, documents, and natural language.

| Task | Description |
|------|-------------|
| [ask](./tasks/ask.md) | Answer a natural language question using indexed project content (RAG) |
| [summarize](./tasks/summarize.md) | Generate a summary of a document or text, with cross-lingual support |
| [embedding](./tasks/embedding.md) | Convert text into a numerical vector for semantic search and comparison |
| [keywords](./tasks/keywords.md) | Extract the most representative keywords or short phrases from a text |
| [key-point](./tasks/key-point.md) | Extract the main ideas from a text as a short list of concise statements |
| [translate](./tasks/translate.md) | Translate a list of texts from one language to another |
| [detect-language](./tasks/detect-language.md) | Identify the language of one or more text samples |
| [entity-extraction](./tasks/entity-extraction.md) | Extract named entities (people, organizations, locations, etc.) from text |
| [search](./tasks/search.md) | Perform semantic search over indexed project content |
| [ingest-content](./tasks/ingest-content.md) | Index document content into the vector database for semantic search |
| [document-extraction](./tasks/document-extraction.md) | Extract and parse text content from uploaded files (PDF, Word, HTML, audio, etc.) |

---

### Dataset Tasks

These tasks analyze structured data stored in datasets.

| Task | Description |
|------|-------------|
| [summary](./tasks/summary.md) | Quick statistical overview of all fields in a dataset |
| [query](./tasks/query.md) | Custom query with filtering, grouping, and aggregation across one or more datasets |
| [chart](./tasks/chart.md) | Generate chart data (bar, line, pie, scatter) from a dataset field |
| [distribution](./tasks/distribution.md) | Analyze how values of a single field are distributed (histogram or frequency) |
| [group-by](./tasks/group-by.md) | Aggregate a numeric field grouped by a categorical field |
| [correlation](./tasks/correlation.md) | Measure the statistical relationship between two numeric fields |
| [correlation-matrix](./tasks/correlation-matrix.md) | Compute correlations between all numeric fields as a heatmap matrix |
| [pivot-table](./tasks/pivot-table.md) | Cross-tabulation of two categorical fields with optional numeric aggregation |
| [time-series](./tasks/time-series.md) | Track how a numeric value changes over time, grouped by month, quarter, or year |
| [outliers](./tasks/outliers.md) | Detect unusual values in a numeric field using the IQR method |
