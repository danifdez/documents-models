# Translate text

Translation converts one or more texts from a source language to a target language.

## Result

Translations preserve the original input order. Each result includes the translated and original text, plus any reference that was associated with the original item.

The source defaults to English and the target defaults to Spanish when the action does not specify them. If several target languages are supplied, only the first is used. Languages use two-letter codes such as `en`, `es`, or `fr`.

Translation uses a separate model for each language pair. If the requested model is unavailable, the action fails explicitly rather than silently using another pair.
