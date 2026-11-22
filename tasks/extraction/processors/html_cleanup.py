_BLOCK_TAGS = ("div", "p", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6")


def normalize_html_blocks(root, extra_block_tags=()):
    """Remove presentation attributes and retain leaf divs as paragraphs."""
    for tag in root.find_all(True):
        for attribute in ("style", "class", "id"):
            tag.attrs.pop(attribute, None)
    for div in root.find_all("div"):
        if not div.find(_BLOCK_TAGS + extra_block_tags):
            div.name = "p"
    for div in root.find_all("div"):
        div.unwrap()
