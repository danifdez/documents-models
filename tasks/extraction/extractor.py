import os
import tempfile

from common.execution_registry import execution_handler
from tasks.extraction.processors.html_processor import process_html
from tasks.extraction.processors.doc_processor import process_doc
from tasks.extraction.processors.pdf_processor import process_pdf
from tasks.extraction.processors.txt_processor import process_txt
from tasks.extraction.processors.eml_processor import process_eml
from tasks.extraction.processors.odt_processor import process_odt
from tasks.extraction.processors.media_processor import process_media
from tasks.extraction.processors.mhtml_processor import process_mhtml


def _materialize(content: bytes, extension: str) -> str:
    fd, path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except Exception:
        os.close(fd)
        if os.path.exists(path):
            os.remove(path)
        raise
    return path


def _extract_by_extension(payload: dict) -> dict:
    ext = payload["extension"]
    content = (payload.get("_input_artifacts") or {}).get("document")
    if content is None:
        return {"error": "extraction step is missing its document artifact"}

    if ext in ['.html', '.htm']:
        html_content = content.decode('utf-8', errors='replace')
        return process_html(html_content)
    if ext in ['.txt', '.md']:
        return process_txt(content.decode('utf-8', errors='replace'))
    # Una página guardada por el navegador: la página y sus adjuntos en un solo
    # fichero. Se lee del blob como el HTML, sin pasar por disco.
    if ext in ['.mhtml', '.mht']:
        return process_mhtml(content)

    tmp_path = _materialize(content, ext)
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


@execution_handler("document-extraction")
def extract(payload) -> dict:
    try:
        return _extract_by_extension(payload)
    except Exception as e:
        return {"error": str(e)}


@execution_handler("indexed-file-extraction")
def extract_indexed_file(payload) -> dict:
    """
    Extracts plain text from an IndexedFile blob.

    Payload keys mirror `document-extraction`:
        - extension (str): file extension including the leading dot.
        - _input_artifacts.document (bytes): file contents.
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
