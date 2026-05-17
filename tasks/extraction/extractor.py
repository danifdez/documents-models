import os
import tempfile

from utils.job_registry import job_handler
from tasks.extraction.processors.html_processor import process_html
from tasks.extraction.processors.doc_processor import process_doc
from tasks.extraction.processors.pdf_processor import process_pdf
from tasks.extraction.processors.txt_processor import process_txt
from tasks.extraction.processors.eml_processor import process_eml
from tasks.extraction.processors.odt_processor import process_odt
from tasks.extraction.processors.media_processor import process_media


def _materialize(input_blob: bytes, extension: str) -> str:
    """Write the job's input_blob to a tempfile so file-based processors can use it."""
    fd, path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(input_blob)
    except Exception:
        os.close(fd)
        if os.path.exists(path):
            os.remove(path)
        raise
    return path


def _extract_by_extension(payload: dict) -> dict:
    ext = payload["extension"]
    input_blob = payload.get("_input_blob")
    if input_blob is None:
        return {"error": "extraction job is missing input_blob"}

    if ext in ['.html', '.htm']:
        html_content = input_blob.decode('utf-8', errors='replace')
        return process_html(html_content)
    if ext in ['.txt', '.md']:
        return process_txt(input_blob.decode('utf-8', errors='replace'))

    tmp_path = _materialize(input_blob, ext)
    try:
        if ext in ['.doc', '.docx']:
            return process_doc(tmp_path)
        if ext in ['.pdf']:
            return process_pdf(tmp_path)
        if ext in ['.eml']:
            return process_eml(tmp_path)
        if ext in ['.odt']:
            return process_odt(tmp_path)
        if ext in ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',
                   '.opus', '.aiff', '.aif',
                   '.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm', '.wmv']:
            return process_media(tmp_path)
        raise ValueError(f"Unsupported file type: {ext}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@job_handler("document-extraction")
def extract(payload) -> dict:
    try:
        return _extract_by_extension(payload)
    except Exception as e:
        return {"error": str(e)}


@job_handler("indexed-file-extraction")
def extract_indexed_file(payload) -> dict:
    """
    Extracts plain text from an IndexedFile blob.

    Payload keys mirror `document-extraction`:
        - extension (str): file extension including the leading dot.
        - _input_blob (bytes): file contents.
        - indexedFileId (int): the IndexedFile id (echoed in the response so the
          backend processor knows which row to update).
    """
    try:
        result = _extract_by_extension(payload)
        if isinstance(result, dict) and "error" not in result:
            result.setdefault("indexedFileId", payload.get("indexedFileId"))
        return result
    except Exception as e:
        return {"error": str(e)}
