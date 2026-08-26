# Text and language features

## Language detection

Documents can identify the language of one or more text samples. Each sample is handled independently and receives a two-letter language code such as `en`, `es`, or `fr`, in the same order as the input.

## Summarization

Long documents are divided into bounded sections. Each section is summarized, then the accepted partial summaries are combined in document order into one final response. The output can use a requested target language.

The feature fails explicitly when its local language model is unavailable or does not produce a valid result.

## Translation

Documents translates one or more texts from a source language to a target language. Results preserve the original order and keep any reference associated with each text.

Translation uses a separate model for each language pair. If the requested pair is unavailable, the action returns an error rather than substituting another language.

## Entity extraction

Entity extraction identifies these categories:

| Category | Includes |
|---|---|
| Person | Real or fictional people. |
| National, religious, or political group | Nationalities and named groups. |
| Facility | Buildings, airports, roads, bridges, and similar places. |
| Organization | Companies, agencies, institutions, and other organizations. |
| Geopolitical place | Countries, cities, and states. |
| Location | Geographic locations such as mountain ranges or bodies of water. |
| Product | Named objects, vehicles, foods, and similar products. |
| Event | Named battles, storms, wars, sporting events, and similar events. |
| Work of art | Titles of books, songs, and other works. |
| Law | Named legal documents. |
| Language | Named languages. |

Large documents are processed in sections. Duplicate names are merged without regard to capitalization while preserving the first appearance. Results remain pending until a user reviews and confirms them in Documents.

## Date extraction

Dates and date ranges are normalized while preserving the original expression, precision, position, and surrounding context. Relative expressions need an anchor date; without one they remain explicitly unresolved. A valid section with no dates produces an empty result, not an error.

## Key points

Documents extracts candidate sentences from each section, removes exact duplicates, and refines them into a short final list. The default final limit is five points. Invalid input, unavailable inference, or empty model output fails explicitly.

## Keywords

Documents extracts concise candidates from each section, counts a candidate at most once per section, compares capitalization-insensitively, and ranks by frequency and first appearance. By default, the result contains up to ten phrases of no more than three words each.

## Dataset statistics

For structured data, Documents can apply equality, inequality, comparison, and text-containment filters before calculating field statistics. Numeric fields include averages, spread, minimum, maximum, and quartiles; text fields include unique and most frequent values; Boolean fields include true and false counts.
