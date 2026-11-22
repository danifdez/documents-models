import logging
import re
from trafilatura import extract
from bs4 import BeautifulSoup

from .html_cleanup import normalize_html_blocks

logger = logging.getLogger(__name__)

_HTML_PARSER = "html.parser"
_author_meta_names = ["author", ":author", "byl", "dc.creator"]
_date_meta_names = [":published_time", ":publishtime", "date", "publication_date","dc.date.issued", "pubdate", "timestamp"]

def _get_meta_content(soup, names):
    meta_tags = soup.find_all("meta", attrs={"content": True})

    for tag in meta_tags:
        for attr in ("name", "property"):
            value = tag.get(attr, "").lower()
            if any(value.endswith(suffix.lower()) for suffix in names):
                return tag["content"]
    return None

def _convert_custom_table_to_html(soup):
    for table in soup.find_all('table'):
        new_table = soup.new_tag('table')
        rows = table.find_all('row')
        for row in rows:
            tr = soup.new_tag('tr')
            cells = row.find_all('cell')
            for cell in cells:
                if cell.get('role') == 'head':
                    th = soup.new_tag('th')
                    th.string = cell.get_text(strip=True)
                    tr.append(th)
                else:
                    td = soup.new_tag('td')
                    td.string = cell.get_text(strip=True)
                    tr.append(td)
            # If no cells, add empty td
            if not cells:
                td = soup.new_tag('td')
                td.string = ''
                tr.append(td)
            new_table.append(tr)
        table.replace_with(new_table)
    return soup

def _clean_extracted_html(extracted):
    soup = BeautifulSoup(extracted, _HTML_PARSER)
    normalize_html_blocks(soup)
    if soup.find("html") and (body := soup.find("body")):
        body_html = "".join(str(content) for content in body.contents)
        soup = BeautifulSoup(body_html, _HTML_PARSER)
    for tag in soup.find_all(["head", "script", "style"]):
        tag.decompose()
    for graphic in soup.find_all("graphic"):
        graphic.name = "img"

    soup = _convert_custom_table_to_html(soup)
    html = re.sub(r"\s+", " ", str(soup))
    html = re.sub(r"<p>\s*</p>", "", html)
    return re.sub(r"</?(?:html|body)>", "", html).strip()


def process_html(html_content) -> dict:
    extracted = extract(
        html_content, output_format="html", favor_precision=True, include_formatting=True, include_links=True, include_images=True, include_tables=True)

    soup = BeautifulSoup(html_content, _HTML_PARSER)
    title = soup.title.string if soup.title else "No Title"
    author = _get_meta_content(soup, _author_meta_names)
    publication_date = _get_meta_content(soup, _date_meta_names)

    if extracted:
        try:
            extracted = _clean_extracted_html(extracted)
        except Exception:
            logger.exception("Error processing HTML content")

    if extracted:
        extracted = extracted.strip()

    return {"content": extracted, "title": title, "author": author, "publication_date": publication_date}
