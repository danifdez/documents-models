import re
import html
from bs4 import BeautifulSoup

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PARAGRAPH_TAGS = {"p", "li", "blockquote", "pre"}
_HTML_TAG_RE = re.compile(r'<[a-zA-Z/!?]')

# Defenses against pathological inputs (data URIs, base64 blobs, minified
# payloads). Any LLM-facing pipeline should pre-process its input through
# `strip_dense_blobs` so a single inline blob can't blow the context window.
_DATA_URI_RE = re.compile(
    r"data:[a-zA-Z0-9+./;=-]*;base64,[A-Za-z0-9+/=\s]+",
    re.MULTILINE,
)
_HUGE_TOKEN_RE = re.compile(r"\S{2000,}")


def strip_dense_blobs(text: str) -> str:
    """Replace data URIs and very long unbroken tokens with short placeholders.

    `data:...;base64,...` becomes `[image]` and any non-whitespace run of
    >=2000 chars becomes `[blob]`. Idempotent and safe to apply multiple
    times.
    """
    if not text:
        return text
    cleaned = _DATA_URI_RE.sub("[image]", text)
    cleaned = _HUGE_TOKEN_RE.sub("[blob]", cleaned)
    return cleaned


def normalize_text(text: str) -> str:
    """Strip HTML tags, unescape HTML entities, and normalize whitespace."""
    try:
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return text


def html_to_markdown(content: str) -> str:
    """Convert HTML to Markdown, preserving headings, lists, links, code, emphasis.

    Used by LLM tasks (summarize, keywords, key-point, relationship-extraction)
    so the model receives structured Markdown rather than tag-stripped plain text.
    Inputs that don't look like HTML are returned unchanged.
    """
    if not content:
        return ""
    if _HTML_TAG_RE.search(content):
        try:
            from markdownify import markdownify as _md
            return _md(content, heading_style="ATX").strip()
        except ImportError:
            return normalize_text(content)
    return content

# Function to extract text from each HTML block element and return as array
def clean_html_text(html_content):
    """
    Extract complete text from each HTML block element, ignoring inline formatting tags.
    """
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    
    block_elements = [
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
        'li', 'td', 'th', 'blockquote', 'pre', 'section', 
        'article', 'header', 'footer', 'main', 'aside'
    ]
    
    text_elements = []
    
    for tag_name in block_elements:
        for element in soup.find_all(tag_name):
            text = element.get_text()
            text = re.sub(r'\s+', ' ', text.strip())
            
            if text:
                text_elements.append(text)
    
    soup_copy = BeautifulSoup(html_content, 'html.parser')
    for script in soup_copy(["script", "style"]):
        script.decompose()
    
    for tag_name in block_elements:
        for element in soup_copy.find_all(tag_name):
            element.decompose()
    
    remaining_text = soup_copy.get_text()
    remaining_text = re.sub(r'\s+', ' ', remaining_text.strip())
    if remaining_text:
        text_elements.append(remaining_text)
    
    return text_elements

# Function to chunk text array into approximately 200 word chunks without splitting elements
def chunk_text(text_elements, words_per_chunk=200):
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for text_element in text_elements:
        element_word_count = len(text_element.split())
        
        if current_word_count + element_word_count > words_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [text_element]
            current_word_count = element_word_count
        else:
            current_chunk.append(text_element)
            current_word_count += element_word_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _recursive_split(text, max_words):
    """Split text recursively: paragraphs -> lines -> sentences -> words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            segments = []
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                # Re-add sentence period if we split on ". "
                if sep == ". " and i < len(parts) - 1:
                    part += "."
                # Recursively split segments that are still too large
                if len(part.split()) > max_words:
                    segments.extend(_recursive_split(part, max_words))
                else:
                    segments.append(part)
            if len(segments) > 1:
                return segments

    # Final fallback: hard split by word count
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def extract_section_units(content: str):
    """Return semantic text units, prioritizing:
    1. Sections (each h1-h6 plus its following paragraphs until the next heading).
    2. Paragraph-like HTML blocks (when no headings are present).
    3. Plain-text paragraphs split by blank lines (when input is not HTML).
    """
    if not content or not content.strip():
        return []

    if "<" in content and ">" in content:
        soup = BeautifulSoup(content, "html.parser")
        for s in soup(["script", "style"]):
            s.decompose()

        ordered = []
        for el in soup.find_all(list(_HEADING_TAGS | _PARAGRAPH_TAGS)):
            text = re.sub(r"\s+", " ", el.get_text()).strip()
            if text:
                ordered.append((el.name in _HEADING_TAGS, text))

        if any(is_heading for is_heading, _ in ordered):
            sections, current = [], []
            for is_heading, text in ordered:
                if is_heading and current:
                    sections.append(" ".join(current))
                    current = []
                current.append(text)
            if current:
                sections.append(" ".join(current))
            sections = [s for s in sections if s.strip()]
            if sections:
                return sections

        paragraphs = clean_html_text(content)
        if paragraphs:
            return paragraphs

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    return paragraphs or [content.strip()]


def chunk_units(units, max_size, size_fn=None, max_words_fallback=None, joiner=" "):
    """Pack units into chunks where the total size_fn cost stays <= max_size per chunk.
    Oversized single units are broken via _recursive_split (paragraphs -> lines -> sentences -> words),
    and the resulting pieces are re-packed up to max_size; any piece still too large is hard-split
    by word count.

    size_fn: callable(str) -> int. Defaults to word count.
    max_words_fallback: max_words passed to _recursive_split for oversized units. Defaults to max_size.
    joiner: how to join units within a chunk.
    """
    if size_fn is None:
        size_fn = lambda s: len(s.split())
    if max_words_fallback is None:
        max_words_fallback = max(50, max_size)

    def _hard_word_split(text):
        words = text.split()
        step = max(1, max_words_fallback)
        return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    def _split_oversized(text):
        pieces = [p.strip() for p in _recursive_split(text, max_words_fallback) if p.strip()]
        packed = []
        cur, cur_size = [], 0
        for piece in pieces:
            psize = size_fn(piece)
            if psize > max_size:
                if cur:
                    packed.append(joiner.join(cur))
                    cur, cur_size = [], 0
                packed.extend(_hard_word_split(piece))
                continue
            if cur_size + psize > max_size and cur:
                packed.append(joiner.join(cur))
                cur = [piece]
                cur_size = psize
            else:
                cur.append(piece)
                cur_size += psize
        if cur:
            packed.append(joiner.join(cur))
        return packed

    chunks = []
    current, current_size = [], 0

    for unit in units:
        unit_size = size_fn(unit)

        if unit_size > max_size:
            if current:
                chunks.append(joiner.join(current))
                current, current_size = [], 0
            chunks.extend(_split_oversized(unit))
            continue

        if current_size + unit_size > max_size and current:
            chunks.append(joiner.join(current))
            current = [unit]
            current_size = unit_size
        else:
            current.append(unit)
            current_size += unit_size

    if current:
        chunks.append(joiner.join(current))

    return chunks


def semantic_chunk_text(text_elements, target_words=None, max_words=None, overlap_words=None):
    """
    Chunk text using recursive splitting with overlap.
    Works well for both small and large texts.
    """
    if target_words is None or max_words is None or overlap_words is None:
        from lib.llm.config import get_rag_config
        rag = get_rag_config()
        if target_words is None:
            target_words = rag.get("chunk_target_words", 150)
        if max_words is None:
            max_words = rag.get("chunk_max_words", 250)
        if overlap_words is None:
            overlap_words = rag.get("chunk_overlap_words", 30)

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
    segments = _recursive_split(full_text, max_words)

    # Accumulate segments into chunks with overlap
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
            if overlap_words > 0 and len(current_words) > overlap_words:
                overlap = current_words[-overlap_words:]
                current_words = overlap + seg_words
                current_count = len(current_words)
            else:
                current_words = seg_words
                current_count = seg_count
        else:
            current_words.extend(seg_words)
            current_count += seg_count

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks