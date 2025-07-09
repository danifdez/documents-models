import os
from utils.job_registry import job_handler
from tasks.extraction.processors.html_processor import process_html
from tasks.extraction.processors.doc_processor import process_doc
from tasks.extraction.processors.pdf_processor import process_pdf
from tasks.extraction.processors.txt_processor import process_txt

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
    extraction_dir = os.getenv("EXTRACTION_DIR", "/app/documents_storage")

    return os.path.join(extraction_dir, hash[:3], hash[3:6],  hash + extension)
