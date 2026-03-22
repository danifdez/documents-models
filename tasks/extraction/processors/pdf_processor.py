from tasks.extraction.processors.docling_processor import process_with_docling


def process_pdf(file) -> dict:
    return process_with_docling(file)
