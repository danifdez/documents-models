## Document Extraction

The **document-extraction** task extracts the text content from an uploaded file and converts it into clean HTML. It supports a variety of file formats including PDFs, Word documents, plain text, emails, and media files.

### What it does

Given a file identified by its hash and extension, the task reads the file from storage, parses it using the appropriate processor, and returns the extracted content as HTML. For media files (audio and video), it performs speech-to-text transcription.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hash` | string | Yes | The file's storage hash (used to locate the file on disk) |
| `extension` | string | Yes | The file extension including the dot (e.g. `.pdf`, `.docx`) |

**Supported extensions:**

| Format | Extensions |
|--------|-----------|
| HTML | `.html`, `.htm` |
| Word | `.doc`, `.docx` |
| PDF | `.pdf` |
| Plain text | `.txt` |
| Email | `.eml` |
| OpenDocument | `.odt` |
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a`, `.wma`, `.opus`, `.aiff`, `.aif` |
| Video | `.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv` |

### Returns

```json
{
  "content": "<p>Extracted text content...</p>"
}
```

For PDFs and Word documents, additional metadata may be included:

```json
{
  "content": "<p>Extracted text...</p>",
  "pages": 12
}
```

For HTML files, metadata extracted from meta tags is included:

```json
{
  "content": "<p>Extracted text...</p>",
  "title": "Document Title",
  "author": "Author Name",
  "publication_date": "2024-01-15"
}
```

On error:

```json
{
  "error": "Unsupported file type: .xyz"
}
```

### Example

**Input:**

```json
{
  "hash": "a1b2c3d4e5f6789abcdef1234567890a",
  "extension": ".pdf"
}
```

**Output:**

```json
{
  "content": "<p>Introduction</p><p>This paper presents a new approach to...</p>",
  "pages": 8
}
```
