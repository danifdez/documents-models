"""Run the native indexing transform over a bounded local JSON request."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.text import clean_html_text, semantic_chunk_text

MAX_REQUEST_BYTES = 512 * 1024


def main():
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Invalid text component input size")
    data = json.loads(raw)
    if set(data) != {"format", "text", "target_words", "max_words", "overlap_words"}:
        raise ValueError("Invalid text component fields")
    kind, text = data["format"], data["text"]
    target, maximum, overlap = (data[key] for key in ("target_words", "max_words", "overlap_words"))
    if (kind not in ("html", "text") or not isinstance(text, str)
            or any(type(number) is not int for number in (target, maximum, overlap))
            or not 1 <= target <= maximum <= 1000 or not 0 <= overlap <= target // 2):
        raise ValueError("Invalid text component options")
    blocks = clean_html_text(text) if kind == "html" else [text]
    chunks = semantic_chunk_text(blocks, target, maximum, overlap)
    print(json.dumps({"blocks": blocks, "chunks": chunks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
