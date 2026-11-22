import os
import mutagen
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

from common.media_formats import VIDEO_EXTENSIONS


_DURATION_LABEL = 'Duración'


_SAMPLE_RATE_LABEL = 'Sample rate'


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def _get_tag(tags, *keys) -> str | None:
    """Try multiple tag keys and return the first non-empty string value."""
    if not tags:
        return None
    for key in keys:
        val = tags.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = val[0] if val else None
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return None


def _extract_id3_fields(audio):
    """Extract title, author, date from ID3 tags (MP3)."""
    tags = audio.tags
    title = _get_tag(tags, 'TIT2', 'TIT1')
    author = _get_tag(tags, 'TPE1', 'TPE2', 'TCOM')
    date = _get_tag(tags, 'TDRC', 'TYER', 'TDAT')
    if date:
        date = str(date)[:10]
    return title, author, date


def _extract_mp4_fields(audio):
    tags = audio.tags or {}
    title = _get_tag(tags, '\xa9nam')
    author = _get_tag(tags, '\xa9ART', 'aART', '\xa9wrt')
    date = _get_tag(tags, '\xa9day')
    if date:
        date = str(date)[:10]
    return title, author, date


def _extract_vorbis_fields(audio):
    tags = audio.tags or {}
    title = _get_tag(tags, 'title', 'TITLE')
    author = _get_tag(tags, 'artist', 'ARTIST', 'author', 'AUTHOR')
    date = _get_tag(tags, 'date', 'DATE', 'year', 'YEAR')
    if date:
        date = str(date)[:10]
    return title, author, date


def _build_content(info_rows: list[tuple[str, str]]) -> str:
    rows_html = ''.join(
        f'<tr><td><strong>{label}</strong></td><td>{value}</td></tr>'
        for label, value in info_rows
    )
    return f'<table>{rows_html}</table>'


_MP4_EXTENSIONS = (".mp4", ".m4v", ".mov", ".m4a", ".aac")


def _read_wave_tags(audio):
    return _extract_id3_fields(audio) if audio.tags else (None, None, None)


def _read_generic_tags(audio):
    tags = audio.tags or {}
    title = _get_tag(tags, "title", "TITLE", "TIT2", "\xa9nam")
    author = _get_tag(tags, "artist", "ARTIST", "TPE1", "\xa9ART")
    date = _get_tag(tags, "date", "DATE", "year", "YEAR", "TDRC", "\xa9day")
    return title, author, str(date)[:10] if date else None


def _read_media(file_path, extension):
    if extension in _MP4_EXTENSIONS:
        return MP4(file_path), _extract_mp4_fields
    parsers = {
        ".mp3": (MP3, _extract_id3_fields),
        ".flac": (FLAC, _extract_vorbis_fields),
        ".ogg": (OggVorbis, _extract_vorbis_fields),
        ".wav": (WAVE, _read_wave_tags),
    }
    parser, read_tags = parsers.get(extension, (mutagen.File, _read_generic_tags))
    return parser(file_path), read_tags


def _mp4_info(info, extension, is_video):
    duration = _format_duration(info.length) if hasattr(info, "length") else "—"
    bitrate = f"{info.bitrate // 1000} kbps" if getattr(info, "bitrate", None) else "—"
    rows = [("Formato", extension.lstrip(".").upper()), (_DURATION_LABEL, duration), ("Bitrate", bitrate)]
    if is_video and getattr(info, "video_fps", None):
        rows.append(("FPS", f"{info.video_fps:.2f}"))
    return rows


def _media_info(audio, extension, is_video):
    if extension in _MP4_EXTENSIONS:
        return _mp4_info(audio.info, extension, is_video)
    if extension in {".mp3", ".flac", ".ogg", ".wav"}:
        return _fixed_audio_info(audio.info, extension)
    rows = [("Formato", extension.lstrip(".").upper())]
    if hasattr(audio, "info") and hasattr(audio.info, "length"):
        rows.append((_DURATION_LABEL, _format_duration(audio.info.length)))
    return rows


def _fixed_audio_info(info, extension):
    label = "OGG Vorbis" if extension == ".ogg" else extension.lstrip(".").upper()
    rows = [("Formato", label), (_DURATION_LABEL, _format_duration(info.length))]
    if extension in {".mp3", ".ogg"}:
        bitrate = "—" if extension == ".ogg" and not info.bitrate else f"{info.bitrate // 1000} kbps"
        rows.append(("Bitrate", bitrate))
    rows.extend([("Canales", str(info.channels)), (_SAMPLE_RATE_LABEL, f"{info.sample_rate} Hz")])
    if extension in {".flac", ".wav"}:
        rows.append(("Bits por muestra", str(info.bits_per_sample)))
    return rows


def process_media(file_path: str) -> dict:
    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    title = None
    author = None
    publication_date = None
    info_rows = []

    try:
        audio, tag_reader = _read_media(file_path, ext)
        if audio is not None:
            title, author, publication_date = tag_reader(audio)
            info_rows = _media_info(audio, ext, is_video)

    except Exception as e:
        info_rows = [('Error', str(e))]

    media_type = 'Vídeo' if is_video else 'Audio'
    if info_rows:
        info_rows.insert(0, ('Tipo', media_type))

    content = _build_content(info_rows) if info_rows else f'<p>{media_type}</p>'

    return {
        'title': title,
        'author': author,
        'publication_date': publication_date,
        'content': content,
    }
