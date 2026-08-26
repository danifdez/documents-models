# Summarize

Summarization creates one concise response from a document or supplied text, in the requested language.

## Large documents

Documents converts the content to plain text and divides it into bounded sections. Each section receives one partial summary. After all required parts are accepted, those summaries are combined in their original order into the final response.

Users see the final summary rather than the internal partial results.

## Languages and failures

The input language can be supplied when known, and the output language is required. Language codes use the two-letter ISO format.

Both the section summaries and the final combination use the configured local language model. If the model is unavailable or produces an invalid result, the action fails explicitly.
