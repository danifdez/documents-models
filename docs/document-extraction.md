# Document Extraction

The `document-extraction` job type extracts text content from uploaded files and normalizes all formats to clean HTML. The extraction pipeline is implemented in `tasks/extraction/`.

## Supported Formats

| Format | Extensions | Library | Processor |
|--------|-----------|---------|-----------|
| PDF | `.pdf` | Docling (StandardPdfPipeline) | `processors/pdf_processor.py` |
| Word | `.doc`, `.docx` | Docling | `processors/doc_processor.py` |
| HTML | `.html`, `.htm` | Trafilatura + BeautifulSoup | `processors/html_processor.py` |
| Plain Text | `.txt` | Built-in | `processors/txt_processor.py` || Email | `.eml` | Python `email` (stdlib) | `processors/eml_processor.py` |
| OpenDocument | `.odt` | odfpy | `processors/odt_processor.py` |
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a`, `.wma`, `.opus`, `.aiff`, `.aif` | mutagen | `processors/media_processor.py` |
| Video | `.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv` | mutagen | `processors/media_processor.py` |
Unsupported extensions return an error.

## File Storage Path

Documents are stored using a content-addressed scheme based on their SHA256 hash:

```
{DOCUMENTS_STORAGE_DIR}/{hash[0:3]}/{hash[3:6]}/{hash}.{extension}
```

For example, a PDF with hash `a1b2c3d4e5...` is stored at:

```
/app/documents_storage/a1b/2c3/a1b2c3d4e5.pdf
```

The `DOCUMENTS_STORAGE_DIR` defaults to `/app/documents_storage` and can be overridden via environment variable.

## Output Format

All processors produce clean HTML with:

- No inline `style` attributes
- No `class` attributes
- No `id` attributes
- No `<script>` or `<style>` tags
- Collapsed whitespace
- Empty `<p>` tags removed
- Unnecessary wrapper `<div>` elements unwrapped or converted to `<p>`

## Processor Details

### PDF Processor

Uses Docling's `DocumentConverter` with `StandardPdfPipeline`:

- **OCR**: Disabled (`do_ocr = False`)
- **Images**: Generated at 2x scale (`images_scale = 2.0`), embedded as base64 in the HTML output
- **HTML export**: Uses `export_to_html(image_mode='embedded')`
- **Post-processing**: Strips attributes, unwraps unnecessary divs, converts leaf divs to `<p>` tags

**Output fields:**

| Field | Description |
|-------|-------------|
| `content` | Clean HTML string |
| `pages` | Page count (from `result.document.pages`) |

### DOC/DOCX Processor

Uses the same Docling pipeline and post-processing as PDF. Docling handles the format conversion internally.

**Output fields:**

| Field | Description |
|-------|-------------|
| `content` | Clean HTML string |
| `pages` | Page count (if available) |

### HTML Processor

Uses Trafilatura for main content extraction, with BeautifulSoup for metadata and cleanup:

1. **Content extraction**: Trafilatura extracts the main content with `favor_precision=True`, preserving formatting, links, images, and tables.
2. **Metadata extraction**: BeautifulSoup parses the original HTML for:
   - `title` from `<title>` tag
   - `author` from meta tags (`author`, `:author`, `byl`, `dc.creator`)
   - `publication_date` from meta tags (`:published_time`, `date`, `pubdate`, etc.)
3. **Table conversion**: Custom `<row>`/`<cell>` table markup (from Trafilatura) is converted to standard `<table>`/`<tr>`/`<td>`/`<th>` HTML.
4. **Cleanup**: Same attribute/div cleanup as other processors, plus removal of `<html>`/`<body>` wrapper tags.

**Output fields:**

| Field | Description |
|-------|-------------|
| `content` | Clean HTML string |
| `title` | Page title |
| `author` | Author name (from meta tags, or `null`) |
| `publication_date` | Publication date (from meta tags, or `null`) |

### Plain Text Processor

Minimal processing — reads the file and wraps each non-empty paragraph in `<p>` tags:

```
Line one text     →   <p>Line one text</p>
                       (empty lines skipped)
Line three text   →   <p>Line three text</p>
```

**Output fields:**

| Field | Description |
|-------|-------------|
| `content` | HTML string with `<p>`-wrapped paragraphs |

### EML Processor

Uses Python's standard library `email` module:

- Prefers the HTML body part when available; falls back to plain text converted to `<p>` tags.
- Extracts metadata from email headers: `subject`, `from`, `to`, `date`.
- Attachment parts are ignored.

**Output fields:**

| Field | Description |
|-------|-------------|
| `content` | HTML body or `<p>`-wrapped plain text |
| `title` | Email subject |
| `author` | Sender address (`From` header) |
| `publication_date` | Parsed send date (ISO format, or `null`) |

### ODT Processor

Uses `odfpy` to parse OpenDocument Text files:

- Paragraphs are mapped to `<p>` tags; heading styles (Heading 1–4) map to `<h1>`–`<h4>`.
- Dublin Core metadata is extracted from the document's meta section.

**Output fields:**

| Field | Description |
|-------|-------------|
| `content` | Clean HTML string |
| `title` | Document title (Dublin Core `dc:title`, or `null`) |
| `author` | Document creator (Dublin Core `dc:creator`, or `null`) |
| `publication_date` | Document date (Dublin Core `dc:date`, or `null`) |

### Media Processor (Audio / Video)

Uses `mutagen` to read container metadata — no transcription is performed.

- Extracts embedded tags (title, artist, album, etc.) from the container.
- Computes duration in human-readable format (`mm:ss` or `h:mm:ss`).
- Returns an HTML summary card rather than transcript content.

**Output fields:**

| Field | Description |
|-------|-------------|
| `content` | HTML summary with title, duration, and available metadata fields |
| `title` | Track/video title from tags, or filename |
| `author` | Artist/composer from tags (or `null`) |

## Error Handling

If extraction fails for any format, the handler returns `{"error": "<message>"}` instead of the normal result. The extractor function (`extract()` in `extractor.py`) catches all exceptions and wraps them in an error response.
