import email
import email.header
import email.utils
from email import policy


def _decode_header(value: str | None) -> str | None:
    if not value:
        return None
    parts = email.header.decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or 'utf-8', errors='replace'))
        else:
            decoded.append(part)
    return ''.join(decoded).strip() or None


def _decode_part(part) -> str | None:
    try:
        return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
    except (AttributeError, LookupError, TypeError, ValueError):
        return None


def _multipart_bodies(message):
    bodies = {"text/html": None, "text/plain": None}
    for part in message.walk():
        content_type = part.get_content_type()
        if content_type not in bodies or "attachment" in str(part.get("Content-Disposition", "")):
            continue
        if bodies[content_type] is None:
            bodies[content_type] = _decode_part(part)
    return bodies["text/html"], bodies["text/plain"]


def _get_body(msg) -> str:
    """Return HTML body if available, otherwise plain text converted to <p> tags."""
    if msg.is_multipart():
        html, text = _multipart_bodies(msg)
    elif msg.get_content_type() == "text/html":
        html, text = _decode_part(msg), None
    else:
        html, text = None, _decode_part(msg)
    if html:
        return html
    return "".join(f"<p>{line}</p>" for line in (text or "").splitlines() if line.strip())


def process_eml(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.compat32)

    subject = _decode_header(msg.get('Subject'))
    from_raw = _decode_header(msg.get('From'))
    to_raw = _decode_header(msg.get('To'))
    date_raw = msg.get('Date')

    # Parse sender name
    author = None
    if from_raw:
        name, addr = email.utils.parseaddr(from_raw)
        author = name.strip() if name.strip() else (addr.strip() or None)

    # Parse date
    publication_date = None
    if date_raw:
        try:
            dt = email.utils.parsedate_to_datetime(date_raw)
            publication_date = dt.date().isoformat()
        except Exception:
            pass

    body = _get_body(msg)

    # Prepend email header block
    header_lines = []
    if from_raw:
        header_lines.append(f'<strong>De:</strong> {from_raw}')
    if to_raw:
        header_lines.append(f'<strong>Para:</strong> {to_raw}')
    if subject:
        header_lines.append(f'<strong>Asunto:</strong> {subject}')
    if date_raw:
        header_lines.append(f'<strong>Fecha:</strong> {date_raw}')

    if header_lines:
        header_html = '<p>' + ' &nbsp;|&nbsp; '.join(header_lines) + '</p>'
        content = header_html + body
    else:
        content = body

    return {
        'title': subject,
        'author': author,
        'publication_date': publication_date,
        'content': content,
    }
