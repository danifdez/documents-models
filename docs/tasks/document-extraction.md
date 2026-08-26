# Extract document content

Extraction converts an uploaded file into clean content that Documents can display and use in later actions.

## Supported content

| Type | Extensions | Result |
|---|---|---|
| PDF | `.pdf` | Clean content, embedded images, and page count. OCR is not performed. |
| Word | `.doc`, `.docx` | Clean content and page count when available. |
| Web page | `.html`, `.htm` | Main content and available title, author, and publication date. |
| Plain text | `.txt` | Non-empty paragraphs in order. |
| Email | `.eml` | Message body and available subject, sender, and date; attachments are ignored. |
| OpenDocument | `.odt` | Paragraphs, headings, and available document metadata. |
| Audio and video | See [Document extraction](../document-extraction.md) | Duration and embedded metadata only. |

For media, speech-to-text is handled by the separate [Transcription](./transcribe.md) feature and can run automatically after extraction when available.

Unsupported, missing, damaged, or unreadable files cause an explicit failure. See [Document extraction](../document-extraction.md) for the complete extension list and format-specific limitations.
