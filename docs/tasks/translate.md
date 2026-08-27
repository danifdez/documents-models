# Translate text

Translation converts one or more texts from a source language to a target language.

## Result

Translations preserve the original input order. Each result includes the translated and original text, plus any reference that was associated with the original item.

The source defaults to English when the action does not specify it. A content translation uses one target language; entity retranslation can request several and returns one result per language in the requested order. Languages use two-letter codes such as `en`, `es`, or `fr`.

Large inputs are split into bounded pieces and batches. Documents records each map and reduction step durably, so retries resume the affected step without repeating the complete translation.

Translation uses a separate model for each language pair. If the requested model is unavailable, the action fails explicitly rather than silently using another pair.
