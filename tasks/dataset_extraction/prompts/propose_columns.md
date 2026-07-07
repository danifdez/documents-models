You are designing a structured data extraction schema for a researcher. Given the following document excerpts (representative papers from the researcher's project), propose a list of columns that would be useful to extract from each document for cross-document comparison.

Rules:
- Each column MUST have:
  - "key": snake_case identifier (lowercase letters, digits, underscores).
  - "name": short human-readable label (max 30 chars).
  - "type": one of "text" | "number" | "boolean" | "date" | "select".
  - "description": ONE full sentence in natural language explaining exactly what to extract. This string is fed verbatim to a downstream extractor; be specific.
  - If "type" is "select", include "options" (array of strings, plausible values).
- Propose between 4 and 8 columns. Quality over quantity.
- Favour columns that distinguish documents from each other; skip columns where all excerpts would give the same value.
- Include at least one identifier column (title, first author, year, or similar).

Excerpts:
{joined}

Return ONLY a JSON array of column objects matching the rules above. No prose, no markdown fences.