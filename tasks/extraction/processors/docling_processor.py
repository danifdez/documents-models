import logging
import re
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from bs4 import BeautifulSoup

from .html_cleanup import normalize_html_blocks

logger = logging.getLogger(__name__)


def _replace_math_elements(soup, body):
    for math_element in body.find_all("math"):
        annotation = math_element.find("annotation", attrs={"encoding": "TeX"})
        latex = (annotation.get_text() if annotation else "").strip()
        if latex:
            span = soup.new_tag("span", attrs={"data-formula": latex})
            span.string = latex
        else:
            span = soup.new_tag("span", attrs={"data-formula-error": "true"})
            span.string = "[Formula not extracted]"
        math_element.replace_with(span)


def _clean_document_body(soup):
    body = soup.body
    if not body:
        return ""
    _replace_math_elements(soup, body)
    normalize_html_blocks(body, ("a",))
    for tag in body.find_all(["script", "style"]):
        tag.decompose()

    html = re.sub(r"\s+", " ", body.decode_contents().strip())
    return re.sub(r"<p>\s*</p>", "", html).strip()


def process_with_docling(file) -> dict:
    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=False,
        generate_picture_images=True,
        images_scale=2.0,
        do_formula_enrichment=True,
    )
    pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

    format_options = {InputFormat.PDF: pdf_format_option}
    converter = DocumentConverter(format_options=format_options)

    result = converter.convert(file)

    page_count = len(result.document.pages) if hasattr(
        result.document, 'pages') else None

    html_content = result.document.export_to_html(image_mode='embedded')
    parsed_html = BeautifulSoup(html_content, "html.parser")

    try:
        processed_content = _clean_document_body(parsed_html)
    except Exception:
        logger.exception("Error processing document content")
        processed_content = parsed_html.body.decode_contents() if parsed_html.body else ""

    result_dict = {"content": processed_content}
    if page_count is not None:
        result_dict["pages"] = page_count

    return result_dict
