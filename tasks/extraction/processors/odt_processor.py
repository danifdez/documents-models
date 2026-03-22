from odf.opendocument import load
from odf import text as odf_text
from odf.namespaces import DCNS


_HEADING_STYLES = {'Heading 1', 'Heading 2', 'Heading 3', 'Heading 4',
                   'Título 1', 'Título 2', 'Título 3', 'Título 4'}

_HEADING_TAG_MAP = {
    'Heading 1': 'h1', 'Heading 2': 'h2', 'Heading 3': 'h3', 'Heading 4': 'h4',
    'Título 1': 'h1', 'Título 2': 'h2', 'Título 3': 'h3', 'Título 4': 'h4',
}


def _get_meta(doc, local_name: str) -> str | None:
    """Read a Dublin Core metadata element from the document."""
    try:
        meta = doc.meta
        for child in meta.childNodes:
            if hasattr(child, 'qname') and child.qname[1] == local_name and child.namespaceURI == DCNS:
                value = child.firstChild.data if child.firstChild else None
                return value.strip() if value else None
    except Exception:
        pass
    return None


def _paragraph_to_html(paragraph) -> str:
    """Convert an odf paragraph/heading node to an HTML string."""
    text_content = ''
    for node in paragraph.childNodes:
        if hasattr(node, 'data'):
            text_content += node.data
        elif hasattr(node, 'childNodes'):
            # Handle text:span and similar inline elements
            for sub in node.childNodes:
                if hasattr(sub, 'data'):
                    text_content += sub.data

    text_content = text_content.strip()
    if not text_content:
        return ''

    style_name = paragraph.getAttribute('stylename') or ''
    tag = _HEADING_TAG_MAP.get(style_name, 'p')
    return f'<{tag}>{text_content}</{tag}>'


def process_odt(file_path: str) -> dict:
    doc = load(file_path)

    title = _get_meta(doc, 'title')
    author = _get_meta(doc, 'creator')
    date_raw = _get_meta(doc, 'date')

    # Truncate date to YYYY-MM-DD if it contains time component
    publication_date = None
    if date_raw:
        publication_date = date_raw[:10] if len(date_raw) >= 10 else date_raw

    # Extract body content
    html_parts = []
    body = doc.text
    for node in body.childNodes:
        tag_name = node.qname[1] if hasattr(node, 'qname') else ''
        if tag_name in ('p', 'h'):
            part = _paragraph_to_html(node)
            if part:
                html_parts.append(part)
        elif tag_name == 'list':
            items = []
            for list_item in node.childNodes:
                item_text = ''
                for p in list_item.childNodes:
                    for n in p.childNodes:
                        if hasattr(n, 'data'):
                            item_text += n.data
                if item_text.strip():
                    items.append(f'<li>{item_text.strip()}</li>')
            if items:
                html_parts.append('<ul>' + ''.join(items) + '</ul>')

    content = ''.join(html_parts)

    return {
        'title': title,
        'author': author,
        'publication_date': publication_date,
        'content': content,
    }
