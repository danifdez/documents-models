# Document extraction

Document extraction turns uploaded files into clean content that Documents can display, search, and use in later actions.

## Supported formats

| Content | Extensions | Extracted result |
|---|---|---|
| PDF | `.pdf` | Text and layout converted to clean content, embedded images, and page count. OCR is not performed. |
| Word | `.doc`, `.docx` | Text and layout converted to clean content, with page count when available. |
| Web page | `.html`, `.htm` | Main content, title, author, and publication date when present. Links, images, and tables are preserved where possible. |
| Plain text | `.txt` | Non-empty paragraphs in their original order. |
| Email | `.eml` | HTML body when present, otherwise plain text; subject, sender, and send date. Attachments are ignored. |
| OpenDocument text | `.odt` | Paragraphs, headings, title, author, and document date when present. |
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a`, `.wma`, `.opus`, `.aiff`, `.aif` | Duration and embedded media metadata. No transcript is produced by extraction itself. |
| Video | `.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv` | Duration and embedded media metadata. No transcript is produced by extraction itself. |

Unsupported extensions cause the extraction action to fail explicitly.

## Cleaned content

Extracted text is normalized into safe, consistent content. Scripts, embedded style definitions, element identifiers, and unnecessary formatting attributes are removed. Whitespace is normalized, empty paragraphs are removed, and useful structure such as paragraphs, headings, links, images, and tables is retained where the source format supports it.

## Format-specific behavior

### PDF and Word

Text, layout, and available images are converted for display. PDF extraction does not use optical character recognition, so image-only scanned pages may not produce readable text.

### HTML

Documents favors the main page content and reads common page metadata for title, author, and publication date. Nonstandard table markup is converted into normal tables where possible.

### Email

The HTML message body is preferred over the plain-text alternative. Attachments are not extracted as part of the email; import them separately if they should become resources.

### Audio and video

Extraction creates a metadata summary, not a speech transcript. Transcription is a separate action and can run automatically after media extraction when that capability is available.

## Failures

If a file is unsupported, unavailable, damaged, or cannot be interpreted, Documents reports the extraction as failed and does not present an incomplete result as successful.
