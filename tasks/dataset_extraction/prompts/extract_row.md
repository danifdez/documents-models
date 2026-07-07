You are extracting structured data from a single document. For each field below, output an object with three keys:
  - "value": the value found in the document, or null if the document does not contain it.
  - "_quote": the verbatim sentence or short passage from the document that supports the value (max 280 characters). Empty string if value is null.
  - "_page": the page number if you can determine it from the text, or null otherwise.

HARD RULES:
- Do NOT use general knowledge outside the document.
- If the document does not explicitly or by clear inference contain the value, set value to null AND _quote to "" AND _page to null.
- Never invent quotes. The _quote must be a verbatim substring of the document content shown below.

{audio_clause}Fields to extract:
{field_block}

Document title: {source_title}
Document content:
---
{document_text}
---

Return a single JSON object with one key per field above (in the same order). Adhere strictly to the provided grammar.