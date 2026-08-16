"""Extrae el HTML de un MHTML y lo procesa como una página normal.

Un MHTML es un multipart/related: la página en una parte y cada hoja de estilo,
imagen o fuente en la suya, codificadas en quoted-printable o base64. Es lo que
manda el navegador al guardar una página (Ctrl+S en ia-browser), y sin esto la
extracción lo rechazaba por extensión y el recurso se quedaba sin contenido: ni
buscable ni troceado en vectores.

Solo interesa la parte de texto. Los adjuntos se quedan donde están, dentro del
fichero guardado, que es lo que permite volver a ver la página tal cual era.
"""

import logging
from email import message_from_bytes
from email.message import Message

from tasks.extraction.processors.html_processor import process_html

logger = logging.getLogger(__name__)


def _decode(part: Message) -> str:
    """El cuerpo de una parte, en texto, respetando su codificación."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _root_html(mensaje: Message) -> str:
    """La parte que es la página, no sus adjuntos.

    Chromium pone la página la primera y la señala con el parámetro «start» de
    la cabecera, pero ninguna de las dos cosas está garantizada: si no hay
    «start» se coge el primer text/html, que es lo que hacen los navegadores al
    abrirlo.
    """
    if not mensaje.is_multipart():
        return _decode(mensaje)

    # get_param devuelve una tupla (charset, idioma, valor) cuando el parámetro
    # viene codificado según la RFC 2231, y una cadena en el caso normal.
    inicio = mensaje.get_param("start")
    if isinstance(inicio, tuple):
        inicio = inicio[2]
    inicio = (inicio or "").strip().strip("<>")

    partes_html = []

    for parte in mensaje.walk():
        if parte.get_content_type() != "text/html":
            continue
        partes_html.append(parte)
        cid = (parte.get("Content-ID") or "").strip().strip("<>")
        localizacion = (parte.get("Content-Location") or "").strip()
        if inicio and inicio in (cid, localizacion):
            return _decode(parte)

    if not partes_html:
        return ""

    # Sin «start», la página es la parte de texto más larga: un MHTML puede
    # traer fragmentos HTML de iframes, y son siempre menores que el documento.
    return max((_decode(p) for p in partes_html), key=len)


def process_mhtml(contenido: bytes) -> dict:
    try:
        mensaje = message_from_bytes(contenido)
    except Exception as e:
        logger.error("MHTML ilegible: %s", e)
        return {"error": f"MHTML ilegible: {e}"}

    html = _root_html(mensaje)
    if not html.strip():
        return {"error": "el MHTML no trae ninguna parte HTML"}

    resultado = process_html(html)

    # El título de la página guardada vale más que el que saque el extractor de
    # un fragmento: el navegador lo mandó porque es el que se veía en la pestaña.
    asunto = mensaje.get("Subject")
    if asunto and not resultado.get("title"):
        resultado["title"] = asunto

    return resultado
