import re
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption, StandardPdfPipeline
from bs4 import BeautifulSoup


def process_doc(file) -> dict:
    pipeline_options = StandardPdfPipeline.get_default_options()
    pipeline_options.do_ocr = False
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

    format_options = {InputFormat.PDF: pdf_format_option}
    converter = DocumentConverter(format_options=format_options)

    result = converter.convert(file)

    page_count = len(result.document.pages) if hasattr(
        result.document, 'pages') else None

    html_content = result.document.export_to_html(image_mode='embedded')
    parsed_html = BeautifulSoup(html_content, "html.parser")

    try:
        body = parsed_html.body
        if body:
            for tag in body.find_all(True):
                if tag.has_attr('style'):
                    del tag['style']
                if tag.has_attr('class'):
                    del tag['class']
                if tag.has_attr('id'):
                    del tag['id']

            for div in body.find_all('div'):
                if not div.find(['div', 'p', 'ul', 'ol', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a']):
                    div.name = 'p'

            body_elements = [el for el in body.children if el.name is not None]
            if len(body_elements) == 1 and body_elements[0].name == 'div':
                body_elements[0].unwrap()

            divs_to_check = list(body.find_all('div'))
            for div in divs_to_check:
                if not div.attrs and len([c for c in div.children if c.name == 'div']) == 1 and len([c for c in div.children if c.name is not None]) == 1:
                    div.unwrap()

            for div in body.find_all('div'):
                div.unwrap()

            for tag in body.find_all(['script', 'style']):
                tag.decompose()

            clean_html = body.decode_contents().strip()
            clean_html = re.sub(r'\s+', ' ', clean_html)
            clean_html = re.sub(r'<p>\s*</p>', '', clean_html)

            processed_content = clean_html.strip()
        else:
            processed_content = ""
    except Exception as e:
        print(f"Error processing PDF content: {str(e)}")
        processed_content = str(
            parsed_html.body.decode_contents() if parsed_html.body else "")

    result_dict = {"content": processed_content}
    if page_count is not None:
        result_dict["pages"] = page_count

    return result_dict
