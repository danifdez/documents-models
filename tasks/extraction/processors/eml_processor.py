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


def _get_body(msg) -> str:
    """Return HTML body if available, otherwise plain text converted to <p> tags."""
    html_part = None
    text_part = None

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get('Content-Disposition', ''))
            if 'attachment' in disp:
                continue
            if ct == 'text/html' and html_part is None:
                charset = part.get_content_charset() or 'utf-8'
                try:
                    html_part = part.get_payload(decode=True).decode(charset, errors='replace')
                except Exception:
                    pass
            elif ct == 'text/plain' and text_part is None:
                charset = part.get_content_charset() or 'utf-8'
                try:
                    text_part = part.get_payload(decode=True).decode(charset, errors='replace')
                except Exception:
                    pass
    else:
        ct = msg.get_content_type()
        charset = msg.get_content_charset() or 'utf-8'
        try:
            raw = msg.get_payload(decode=True).decode(charset, errors='replace')
        except Exception:
            raw = ''
        if ct == 'text/html':
            html_part = raw
        else:
            text_part = raw

    if html_part:
        return html_part

    if text_part:
        lines = text_part.splitlines()
        paragraphs = [f'<p>{line}</p>' for line in lines if line.strip()]
        return ''.join(paragraphs)

    return ''


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
