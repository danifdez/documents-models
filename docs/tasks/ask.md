# Ask a question

Ask answers a natural-language question from content already indexed in the current project.

## What happens

Documents selects a bounded, project-scoped set of candidate passages. The processing service ranks those passages by meaning, removes duplicates, adds any relevant project relationships supplied by Documents, and generates one grounded answer.

The processing service cannot search another project, the application database, or the web.

## Result

The result is a natural-language response in the language of the question. If the selected content does not contain relevant information, the response states that no relevant information was found.

There is no heuristic fallback when the language model is unavailable; the action fails explicitly.

## Good use

Ask focused questions whose answer is likely to appear in the imported material, such as “What conclusions does the 2023 annual report reach?” Review important answers against their source documents.
