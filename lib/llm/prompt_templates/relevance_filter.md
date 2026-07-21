You are filtering sections of a document for a {task_label} task. Keep only sections that contain the document's substantive main content. Discard sections that are auxiliary, such as: appendices, annexes, bibliographies, references, citations, footnotes, acknowledgements, author biographies, glossaries, indexes, nomenclature, copyright/license/disclaimer boilerplate, table-of-contents listings, publication metadata, errata, and tables of raw auxiliary data that do not contribute to the main argument.

Sections (each line is one section, prefixed by its index):
{listing}

Respond with a single JSON object of the form {{"keep": [indices]}} listing the indices of sections to retain. Output JSON only — no prose, no markdown fences.