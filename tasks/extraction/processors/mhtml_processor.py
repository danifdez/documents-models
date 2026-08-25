"""Extract the root HTML document from an MHTML archive."""

import logging
from email import message_from_bytes
from email.message import Message

from tasks.extraction.processors.html_processor import process_html

logger = logging.getLogger(__name__)


def _decode(part: Message) -> str:
    """Decode one MIME part using its declared charset."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _root_html(message: Message) -> str:
    if not message.is_multipart():
        return _decode(message) if message.get_content_type() == "text/html" else ""

    start = message.get_param("start")
    if isinstance(start, tuple):
        start = start[2]
    start = (start or "").strip().strip("<>")

    html_parts = []
    for part in message.walk():
        if part.get_content_type() != "text/html":
            continue
        html_parts.append(part)
        content_id = (part.get("Content-ID") or "").strip().strip("<>")
        location = (part.get("Content-Location") or "").strip()
        if start and start in (content_id, location):
            return _decode(part)

    if not html_parts:
        return ""

    return max((_decode(part) for part in html_parts), key=len)


def process_mhtml(content: bytes) -> dict:
    try:
        message = message_from_bytes(content)
    except Exception as exc:
        logger.error("MHTML content could not be parsed: %s", exc)
        raise ValueError("MHTML content could not be parsed") from exc

    html = _root_html(message)
    if not html.strip():
        raise ValueError("MHTML content does not contain an HTML document")

    result = process_html(html)

    subject = message.get("Subject")
    if subject and not result.get("title"):
        result["title"] = subject

    return result
