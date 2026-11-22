from bs4 import BeautifulSoup, Comment, Declaration, Doctype, NavigableString, ProcessingInstruction


BLOCK_ELEMENTS = (
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "td", "th", "blockquote", "pre", "section",
    "article", "header", "footer", "main", "aside",
)


def clean_html_text(html_content):
    """Collect visible text in document order, once per contiguous block."""
    soup = BeautifulSoup(html_content or "", "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()

    blocks = []
    pieces = []
    current = None
    for node in soup.descendants:
        if getattr(node, "name", None) == "br":
            text = " "
        elif isinstance(node, NavigableString) and not isinstance(
            node, (Comment, Doctype, Declaration, ProcessingInstruction)
        ):
            text = str(node)
        else:
            continue
        block = node.find_parent(BLOCK_ELEMENTS)
        if block is not current:
            value = " ".join("".join(pieces).split())
            if value:
                blocks.append(value)
            pieces.clear()
            current = block
        pieces.append(text)
    value = " ".join("".join(pieces).split())
    if value:
        blocks.append(value)
    return blocks


def _recursive_split(text, max_words):
    """Split text recursively: paragraphs -> lines -> sentences -> words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    for separator in ("\n\n", "\n", ". ", " "):
        parts = text.split(separator)
        if len(parts) <= 1:
            continue
        segments = _split_parts(parts, separator, max_words)
        if len(segments) > 1:
            return segments
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _split_parts(parts, separator, max_words):
    segments = []
    for index, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if separator == ". " and index < len(parts) - 1:
            part += "."
        segments.extend(_recursive_split(part, max_words))
    return segments


def _chunk_limits(target_words, max_words, overlap_words):
    if target_words is None or max_words is None or overlap_words is None:
        from lib.llm.config import get_rag_config
        rag = get_rag_config()
        if target_words is None:
            target_words = rag.get("chunk_target_words", 150)
        if max_words is None:
            max_words = rag.get("chunk_max_words", 250)
        if overlap_words is None:
            overlap_words = rag.get("chunk_overlap_words", 30)
    return target_words, max_words, overlap_words


def valid_chunk_limits(target, maximum, overlap):
    return (
        all(type(limit) is int for limit in (target, maximum, overlap))
        and 1 <= target <= maximum
        and 0 <= overlap <= target // 2
    )


def semantic_chunk_text(text_elements, target_words=None, max_words=None, overlap_words=None):
    """
    Chunk text using recursive splitting with overlap.
    Works well for both small and large texts.
    """
    if not isinstance(text_elements, (list, tuple)) or any(not isinstance(text, str) for text in text_elements):
        raise ValueError("Chunk input requires a sequence of text blocks")
    target_words, max_words, overlap_words = _chunk_limits(target_words, max_words, overlap_words)

    if not valid_chunk_limits(target_words, max_words, overlap_words):
        raise ValueError("Invalid chunk word limits")

    if not text_elements:
        return []

    # Join block elements with paragraph separators
    full_text = "\n\n".join(el.strip() for el in text_elements if el.strip())
    if not full_text:
        return []

    total_words = len(full_text.split())

    # Small text: return as single chunk
    if total_words <= max_words:
        return [full_text]

    # Recursively split into semantic segments
    segments = _recursive_split(full_text, max_words - overlap_words)

    return _assemble_chunks(segments, target_words, overlap_words)


def _assemble_chunks(segments, target_words, overlap_words):
    chunks = []
    current_words = []
    current_count = 0

    for segment in segments:
        seg_words = segment.split()
        seg_count = len(seg_words)

        if current_count + seg_count > target_words and current_words:
            # Close current chunk
            chunks.append(" ".join(current_words))
            # Start new chunk with overlap from end of previous
            # Never copy an entire previous chunk, including short paragraphs.
            overlap_size = min(overlap_words, len(current_words) - 1)
            overlap = current_words[-overlap_size:] if overlap_size > 0 else []
            current_words = overlap + seg_words
            current_count = len(current_words)
        else:
            current_words.extend(seg_words)
            current_count += seg_count

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
