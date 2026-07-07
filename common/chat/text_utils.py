"""Small text helpers shared across the assistant_chat package."""

import re


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
