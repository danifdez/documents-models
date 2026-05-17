"""File-format writers for `folder_write`.

The assistant chat tool `folder_write` accepts a single `content` string and a
target filename. This module decides what to do with that content based on the
filename's extension:

- TEXT extensions: write bytes = content.encode('utf-8').
- DOCUMENT extensions (.pdf .docx .odt): treat content as markdown source and
  convert via pandoc (pypandoc-binary bundles the binary). For .pdf the chain
  is `pandoc md -> typst` (intermediate format) followed by `typst.compile()`
  in Python (the `typst` pip package exposes the compiler as a Python module;
  there is no `typst` CLI on PATH).
- SPREADSHEET (.xlsx): treat content as CSV and convert via openpyxl.

Any other extension raises UnsupportedExtension.

All conversion errors are wrapped in ConversionError with a short `reason`
string so the assistant_chat worker can pass them back to the LLM as a
machine-readable error code.
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from typing import Tuple


class UnsupportedExtension(Exception):
    def __init__(self, ext: str):
        self.ext = ext
        super().__init__(f"unsupported_extension:{ext}")


class ConversionError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


_TEXT_EXTENSIONS = {
    # Data / config
    ".txt", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".log",
    # Markup
    ".md", ".html", ".htm", ".svg",
    # Source code
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".bash", ".sql",
    ".css", ".scss", ".less",
    ".go", ".rs", ".rb", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".php", ".r", ".kt", ".swift",
    ".dockerfile",
}

_MD_BINARY_EXTENSIONS = {".pdf", ".docx", ".odt"}
_CSV_XLSX_EXTENSIONS = {".xlsx"}


def normalize_and_categorize(raw_filename: str) -> Tuple[str, str]:
    """Return (normalized_filename, category).

    Rules:
    - Strips surrounding whitespace.
    - `.markdown` -> `.md`.
    - A bare name with no dot in the last path segment defaults to `.md`.
    - Categories: "text" | "md-binary" | "csv-xlsx".
    - Raises UnsupportedExtension for any other extension or empty input.
    """
    name = (raw_filename or "").strip()
    if not name:
        raise UnsupportedExtension("")

    lower = name.lower()
    if lower.endswith(".markdown"):
        name = name[: -len(".markdown")] + ".md"
        lower = name.lower()

    last_segment = lower.rsplit("/", 1)[-1]
    if "." not in last_segment:
        # bare name → default to .md
        return name + ".md", "text"

    ext = "." + last_segment.rsplit(".", 1)[-1]
    if ext in _TEXT_EXTENSIONS:
        return name, "text"
    if ext in _MD_BINARY_EXTENSIONS:
        return name, "md-binary"
    if ext in _CSV_XLSX_EXTENSIONS:
        return name, "csv-xlsx"
    raise UnsupportedExtension(ext)


def to_bytes(content: str, filename: str, category: str) -> bytes:
    """Render `content` to the bytes that should be written to disk."""
    if category == "text":
        return _to_text_bytes(content)
    if category == "md-binary":
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        return _md_to_binary_bytes(content, ext)
    if category == "csv-xlsx":
        return _csv_to_xlsx_bytes(content)
    raise ConversionError("unknown_category", category)


def _to_text_bytes(content: str) -> bytes:
    if content is None:
        return b""
    return content.encode("utf-8")


def _md_to_binary_bytes(md: str, target_ext: str) -> bytes:
    """Convert markdown to one of .pdf/.docx/.odt via pandoc.

    Raises ConversionError on any failure with a short, model-readable
    reason.
    """
    try:
        import pypandoc
    except ImportError as e:
        raise ConversionError("pandoc_not_available", str(e))

    md = md or ""

    if target_ext == ".pdf":
        return _md_to_pdf_bytes(md, pypandoc)

    # .docx / .odt: pandoc writes binary outputs through a file.
    pandoc_format = target_ext.lstrip(".")
    with tempfile.NamedTemporaryFile(suffix=target_ext, delete=False) as f:
        out_path = f.name
    try:
        try:
            pypandoc.convert_text(
                md, pandoc_format, format="md", outputfile=out_path,
            )
        except Exception as e:
            raise ConversionError("conversion_failed", str(e)) from e
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _md_to_pdf_bytes(md: str, pypandoc) -> bytes:
    """Two-step md → typst (via pandoc) → pdf (via typst Python API)."""
    try:
        import typst
    except ImportError as e:
        raise ConversionError("typst_not_available", str(e))

    try:
        typst_source = pypandoc.convert_text(md, "typst", format="md")
    except Exception as e:
        raise ConversionError("conversion_failed", f"md→typst: {e}") from e

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".typ", delete=False, encoding="utf-8",
    ) as f:
        f.write(typst_source)
        typ_path = f.name
    pdf_path = typ_path[:-4] + ".pdf"
    try:
        try:
            typst.compile(typ_path, output=pdf_path)
        except Exception as e:
            raise ConversionError("conversion_failed", f"typst→pdf: {e}") from e
        with open(pdf_path, "rb") as f:
            return f.read()
    finally:
        for p in (typ_path, pdf_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _csv_to_xlsx_bytes(csv_text: str) -> bytes:
    """Convert CSV text (RFC 4180-ish, first row = headers) to xlsx bytes."""
    try:
        import openpyxl
    except ImportError as e:
        raise ConversionError("openpyxl_not_available", str(e))

    try:
        reader = csv.reader(io.StringIO(csv_text or ""))
        rows = list(reader)
    except Exception as e:
        raise ConversionError("csv_parse_error", str(e)) from e

    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append([_coerce_cell(c) for c in row])

    buf = io.BytesIO()
    try:
        wb.save(buf)
    except Exception as e:
        raise ConversionError("conversion_failed", f"openpyxl save: {e}") from e
    return buf.getvalue()


def _coerce_cell(value: str):
    """Best-effort numeric coercion. Strings that parse as int/float become
    numbers (Excel treats them as numeric cells). Anything else stays string."""
    if value is None:
        return ""
    s = value.strip()
    if not s:
        return ""
    # Preserve leading zeros (phone numbers, ZIP codes) as strings.
    if len(s) > 1 and s.startswith("0") and not s.startswith("0."):
        return value
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return value
