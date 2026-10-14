"""Accepted audio and video formats for extraction and transcription.

Single source of truth: the extractor routes these formats to the media
processor and `transcribe` accepts the same set, so neither may declare its own
copy. The user-facing tables in docs/tasks/transcribe.md and
docs/document-extraction.md mirror this module.
"""

AUDIO_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma",
    ".opus", ".aiff", ".aif",
})
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv",
})
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
