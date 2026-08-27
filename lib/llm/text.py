"""Text helpers shared across chat packages and content tasks."""

import re


_DATA_URI_RE = re.compile(
    r"data:[a-zA-Z0-9+./;=-]*;base64,[A-Za-z0-9+/=\s]+",
    re.MULTILINE,
)
_HUGE_TOKEN_RE = re.compile(r"\S{2000,}")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove completed and truncated thinking blocks from a model reply."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in cleaned.lower():
        head = _UNCLOSED_THINK_RE.sub("", cleaned)
        cleaned = head if head.strip() else cleaned
    return cleaned.strip()


def strip_dense_blobs(text: str) -> str:
    """Replace data URIs and very long unbroken tokens with placeholders."""
    if not text:
        return text
    cleaned = _DATA_URI_RE.sub("[image]", text)
    return _HUGE_TOKEN_RE.sub("[blob]", cleaned)


def char_budget(cfg, *, tokens_key="chunk_max_tokens", default_tokens=400):
    """Approximate the input characters available in the model context."""
    override = cfg.get("input_char_budget")
    if override is not None:
        return int(override)
    from lib.llm.config import get_llm_defaults

    n_ctx = int(get_llm_defaults().get("n_ctx", 32768))
    out_tokens = int(cfg.get(tokens_key, default_tokens))
    available_tokens = max(512, n_ctx - out_tokens - 512)
    return available_tokens * 4


def truncate_for_llm(
    text,
    cfg,
    *,
    tokens_key="chunk_max_tokens",
    default_tokens=400,
):
    """Truncate text to the character budget derived from the model context."""
    cap = char_budget(cfg, tokens_key=tokens_key, default_tokens=default_tokens)
    return text if len(text) <= cap else text[:cap]
