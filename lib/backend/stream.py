"""Streaming reply helper shared by the chat task handlers.

A chat turn ends by streaming the model's reply to the UI as it is produced. The
mechanism is identical for the personal assistant and for user-created agents:
iterate `llm.chat_stream`, filter out any `<think>…</think>` block so the caret
only shows user-visible text, flush partial chunks to the backend on a time
budget, and return the raw (unfiltered) text so the caller can strip thinking
once for the persisted reply.

This lives in `lib.backend` because it owns the backend transport (it POSTs
chunks via `post_stream_chunk`); it takes a duck-typed `llm` object and knows
nothing about tasks or agents.
"""

import logging
import time
from typing import Any, Dict, List

from lib.backend.http import post_stream_chunk

logger = logging.getLogger(__name__)

# Flush a partial chunk to the backend roughly every N ms, regardless of how
# many tokens have accumulated. Tuned so the user sees forward motion without
# drowning the HTTP loop in tiny requests.
STREAM_FLUSH_INTERVAL_MS = 120


class _ThinkFilter:
    """Streaming state machine that strips `<think>…</think>` spans from a token
    stream. Even with `/no_think`, Qwen3 sometimes opens an empty pair at the
    start; we forward to the UI only the text outside any thinking block, while
    the caller keeps the raw stream verbatim for the final reply."""

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""

    def feed(self, piece: str) -> str:
        """Return the user-visible portion of this chunk given current thinking
        state. Buffers a partial `<think>` prefix across calls so a tag split
        over two chunks is never emitted."""
        out_parts: List[str] = []
        buf = self._pending + piece
        while buf:
            if self._in_think:
                end = buf.find("</think>")
                if end == -1:
                    buf = ""
                    break
                buf = buf[end + len("</think>"):]
                self._in_think = False
            else:
                start = buf.find("<think>")
                if start == -1:
                    for i in range(1, min(len("<think>"), len(buf)) + 1):
                        if "<think>".startswith(buf[-i:]):
                            out_parts.append(buf[:-i])
                            self._pending = buf[-i:]
                            return "".join(out_parts)
                    out_parts.append(buf)
                    self._pending = ""
                    return "".join(out_parts)
                out_parts.append(buf[:start])
                buf = buf[start + len("<think>"):]
                self._in_think = True
        self._pending = ""
        return "".join(out_parts)


def generate_reply(
    llm: Any,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    *,
    owner_segment: str,
    owner_id: Any,
    job_id: Any,
    stream_enabled: bool = True,
) -> str:
    """Produce the model's reply, streaming it live to the UI when possible.

    Streams (posting chunks to `/:segment/:id/stream-chunk`) when `stream_enabled`
    and both `owner_id`/`job_id` are ints; otherwise falls back to a single
    blocking `chat` call. Returns the RAW text (thinking blocks NOT stripped) —
    the caller strips once for the persisted reply."""
    can_stream = (
        stream_enabled
        and isinstance(owner_id, int)
        and isinstance(job_id, int)
    )
    if not can_stream:
        logger.warning(
            "chat: streaming disabled (owner=%s/%s job=%r stream=%r)",
            owner_segment, owner_id, job_id, stream_enabled,
        )
        return llm.chat(messages, max_tokens=max_tokens, allow_thinking=True) or ""

    logger.info("chat: streaming (owner=%s/%s job=%s)", owner_segment, owner_id, job_id)
    think = _ThinkFilter()
    raw_parts: List[str] = []
    buffer: List[str] = []
    chunks_sent = 0
    last_flush_ms = time.monotonic() * 1000

    for piece in llm.chat_stream(messages, max_tokens=max_tokens):
        raw_parts.append(piece)
        visible = think.feed(piece)
        if visible:
            buffer.append(visible)
        now_ms = time.monotonic() * 1000
        if buffer and now_ms - last_flush_ms >= STREAM_FLUSH_INTERVAL_MS:
            post_stream_chunk(owner_segment, owner_id, job_id, "".join(buffer))
            chunks_sent += 1
            buffer.clear()
            last_flush_ms = now_ms
    if buffer:
        post_stream_chunk(owner_segment, owner_id, job_id, "".join(buffer))
        chunks_sent += 1
    post_stream_chunk(owner_segment, owner_id, job_id, "", done=True)
    logger.info("chat: stream done, %d chunks sent", chunks_sent)
    return "".join(raw_parts)
