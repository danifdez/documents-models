from utils.job_registry import job_handler
from tasks.extraction.processors.html_processor import process_html
from tasks.extraction.processors.doc_processor import process_doc
from tasks.extraction.processors.pdf_processor import process_pdf
from tasks.extraction.processors.txt_processor import process_txt
from tasks.extraction.processors.eml_processor import process_eml
from tasks.extraction.processors.odt_processor import process_odt
from tasks.extraction.processors.media_processor import process_media
from config import DOCUMENTS_STORAGE_DIR
import os


@job_handler("document-extraction")
def extract(payload) -> dict:
    try:
        ext = payload["extension"]

        if ext in ['.html', '.htm']:
            with open(_get_file_path(payload["hash"], ext), 'r', encoding='utf-8') as file:
                html_content = file.read()
            return process_html(html_content)
        elif ext in ['.doc', '.docx']:
            return process_doc(_get_file_path(payload["hash"], ext))
        elif ext in ['.pdf']:
            return process_pdf(_get_file_path(payload["hash"], ext))
        elif ext in ['.txt']:
            return process_txt(_get_file_path(payload["hash"], ext))
        elif ext in ['.eml']:
            return process_eml(_get_file_path(payload["hash"], ext))
        elif ext in ['.odt']:
            return process_odt(_get_file_path(payload["hash"], ext))
        elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.opus', '.aiff', '.aif',
                     '.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm', '.wmv']:
            return process_media(_get_file_path(payload["hash"], ext))
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        return {"error": str(e)}


def _get_file_path(hash: str, extension: str) -> str:
    """
    Constructs the file path for the extracted document based on its hash and extension.

    Args:
        hash (str): The hash of the document.
        extension (str): The file extension of the document.

    Returns:
        str: The absolute path to the file.
    """
    return os.path.join(DOCUMENTS_STORAGE_DIR, hash[:3], hash[3:6],  hash + extension)
