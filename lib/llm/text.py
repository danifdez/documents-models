"""Text helpers shared across the chat packages and the content tasks.

Two groups:

- `strip_thinking`: post-processing of a chat reply.
- The rest: the input pipeline of the agentic content tasks (summarize,
  keywords, key-point, date-extraction). They clean a document
  (HTML → markdown, drop dense blobs), split it into semantic units and pack
  those units into chunks that fit the LLM context.

Mirrors `documents-dev/models/services/text.py`, minus the RAG-only chunkers
(`chunk_text`, `semantic_chunk_text`) which no task ported here needs.
"""

import html
import re

from bs4 import BeautifulSoup

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PARAGRAPH_TAGS = {"p", "li", "blockquote", "pre"}
_HTML_TAG_RE = re.compile(r'<[a-zA-Z/!?]')
# Markdown ATX heading (`# `, `## `, …). The content tasks feed this module the
# output of `html_to_markdown`, so headings arrive as markdown, not HTML tags.
_MD_HEADING_RE = re.compile(r'^#{1,6}\s+')

# Defenses against pathological inputs (data URIs, base64 blobs, minified
# payloads). Any LLM-facing pipeline should pre-process its input through
# `strip_dense_blobs` so a single inline blob can't blow the context window.
_DATA_URI_RE = re.compile(
    r"data:[a-zA-Z0-9+./;=-]*;base64,[A-Za-z0-9+/=\s]+",
    re.MULTILINE,
)
_HUGE_TOKEN_RE = re.compile(r"\S{2000,}")


# Thinking models (Qwen3, DeepSeek-R1, etc.) emit their reasoning chain
# inside <think>...</think>. The user only wants to see the final response.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks. Also drops an unclosed leading <think>
    block (happens if max_tokens cuts the reasoning off mid-stream)."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # If a <think> remains, it never closed — drop everything from it onwards,
    # but only if we still have content before it.
    if "<think>" in cleaned.lower():
        head = _UNCLOSED_THINK_RE.sub("", cleaned)
        cleaned = head if head.strip() else cleaned
    return cleaned.strip()


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


def word_count(text: str) -> int:
    """Count whitespace-separated words. Returns 0 for empty/None input."""
    return len(text.split()) if text else 0


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
                if word_count(part) > max_words:
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

    blocks = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if not blocks:
        return [content.strip()] if content.strip() else []

    # Markdown headings: group each heading with the blocks that follow it into
    # one section (mirroring the HTML branch above), so a heading-based relevance
    # drop takes its body with it. Without this the heading and its paragraphs
    # become separate units and only an LLM reading the content could pair them.
    if any(_MD_HEADING_RE.match(b) for b in blocks):
        sections, current = [], []
        for b in blocks:
            if _MD_HEADING_RE.match(b) and current:
                sections.append("\n\n".join(current))
                current = []
            current.append(b)
        if current:
            sections.append("\n\n".join(current))
        return sections

    return blocks


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
        size_fn = word_count
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


def char_budget(cfg):
    """Approximate max chars that fit alongside the prompt in the LLM context.

    Uses ~4 chars/token (English heuristic) and reserves room for the system
    prompt and output. Honours overrides via `<task>.input_char_budget`.
    """
    override = cfg.get("input_char_budget")
    if override is not None:
        return int(override)
    from lib.llm.config import get_llm_defaults

    n_ctx = int(get_llm_defaults().get("n_ctx", 32768))
    out_tokens = int(cfg.get("chunk_max_tokens", 400))
    # Leave 512 tokens of headroom for the prompt boilerplate.
    available_tokens = max(512, n_ctx - out_tokens - 512)
    return available_tokens * 4


def truncate_for_llm(text, cfg):
    """Truncate `text` to the character budget from `char_budget`."""
    cap = char_budget(cfg)
    if len(text) <= cap:
        return text
    return text[:cap]


def build_chunks(content, chunk_word_budget, *, units_filter=None):
    """Input pipeline of the content tasks: clean the document (HTML → markdown,
    drop dense blobs), split it into semantic units, apply an optional relevance
    filter and pack the units into chunks that fit `chunk_word_budget` words.
    """
    cleaned = strip_dense_blobs(html_to_markdown(content or ""))
    units = extract_section_units(cleaned)
    if not units:
        return []
    if units_filter is not None:
        units = units_filter(units) or units
    return chunk_units(units, chunk_word_budget, joiner="\n\n")
