import os
import mutagen
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE
from mutagen.aiff import AIFF


_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.opus', '.aiff', '.aif'}
_VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm', '.wmv'}


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


def process_media(file_path: str) -> dict:
    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in _VIDEO_EXTENSIONS

    title = None
    author = None
    publication_date = None
    info_rows = []

    try:
        if ext == '.mp3':
            audio = MP3(file_path)
            title, author, publication_date = _extract_id3_fields(audio)
            info = audio.info
            info_rows = [
                ('Formato', 'MP3'),
                ('Duración', _format_duration(info.length)),
                ('Bitrate', f'{info.bitrate // 1000} kbps'),
                ('Canales', str(info.channels)),
                ('Sample rate', f'{info.sample_rate} Hz'),
            ]

        elif ext in ('.mp4', '.m4v', '.mov', '.m4a', '.aac'):
            audio = MP4(file_path)
            title, author, publication_date = _extract_mp4_fields(audio)
            info = audio.info
            duration = _format_duration(info.length) if hasattr(info, 'length') else '—'
            bitrate = f'{info.bitrate // 1000} kbps' if hasattr(info, 'bitrate') and info.bitrate else '—'
            info_rows = [
                ('Formato', ext.lstrip('.').upper()),
                ('Duración', duration),
                ('Bitrate', bitrate),
            ]
            if is_video and hasattr(info, 'video_fps') and info.video_fps:
                info_rows.append(('FPS', f'{info.video_fps:.2f}'))

        elif ext == '.flac':
            audio = FLAC(file_path)
            title, author, publication_date = _extract_vorbis_fields(audio)
            info = audio.info
            info_rows = [
                ('Formato', 'FLAC'),
                ('Duración', _format_duration(info.length)),
                ('Canales', str(info.channels)),
                ('Sample rate', f'{info.sample_rate} Hz'),
                ('Bits por muestra', str(info.bits_per_sample)),
            ]

        elif ext == '.ogg':
            audio = OggVorbis(file_path)
            title, author, publication_date = _extract_vorbis_fields(audio)
            info = audio.info
            info_rows = [
                ('Formato', 'OGG Vorbis'),
                ('Duración', _format_duration(info.length)),
                ('Bitrate', f'{info.bitrate // 1000} kbps' if info.bitrate else '—'),
                ('Canales', str(info.channels)),
                ('Sample rate', f'{info.sample_rate} Hz'),
            ]

        elif ext in ('.wav',):
            audio = WAVE(file_path)
            info = audio.info
            if audio.tags:
                title, author, publication_date = _extract_id3_fields(audio)
            info_rows = [
                ('Formato', 'WAV'),
                ('Duración', _format_duration(info.length)),
                ('Canales', str(info.channels)),
                ('Sample rate', f'{info.sample_rate} Hz'),
                ('Bits por muestra', str(info.bits_per_sample)),
            ]

        else:
            # Generic fallback via mutagen.File
            audio = mutagen.File(file_path)
            if audio is not None:
                tags = audio.tags or {}
                title = _get_tag(tags, 'title', 'TITLE', 'TIT2', '\xa9nam')
                author = _get_tag(tags, 'artist', 'ARTIST', 'TPE1', '\xa9ART')
                publication_date = _get_tag(tags, 'date', 'DATE', 'year', 'YEAR', 'TDRC', '\xa9day')
                if publication_date:
                    publication_date = str(publication_date)[:10]
                if hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                    info_rows = [
                        ('Formato', ext.lstrip('.').upper()),
                        ('Duración', _format_duration(audio.info.length)),
                    ]
                else:
                    info_rows = [('Formato', ext.lstrip('.').upper())]

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
