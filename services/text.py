import re
import html
from bs4 import BeautifulSoup
from config import RAG_CHUNK_TARGET_WORDS, RAG_CHUNK_MAX_WORDS, RAG_CHUNK_OVERLAP_WORDS


def normalize_text(text: str) -> str:
    """Strip HTML tags, unescape HTML entities, and normalize whitespace."""
    try:
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return text

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


def semantic_chunk_text(text_elements, target_words=None, max_words=None, overlap_words=None):
    """
    Chunk text using recursive splitting with overlap.
    Works well for both small and large texts.
    """
    if target_words is None:
        target_words = RAG_CHUNK_TARGET_WORDS
    if max_words is None:
        max_words = RAG_CHUNK_MAX_WORDS
    if overlap_words is None:
        overlap_words = RAG_CHUNK_OVERLAP_WORDS

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