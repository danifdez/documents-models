import os
import logging
import subprocess
import tempfile
from threading import Lock

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.opus', '.aiff', '.aif'}
_VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm', '.wmv'}

_model = None
_model_lock = Lock()
_inference_lock = Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                config = get_task_config("transcribe")
                model_size = config.get("model", "base")
                device = config.get("device", "cpu")
                compute_type = config.get("compute_type", "int8")

                logger.info(
                    "Loading Whisper model: %s "
                    "(device=%s, compute_type=%s)",
                    model_size,
                    device,
                    compute_type,
                )
                _model = WhisperModel(
                    model_size, device=device, compute_type=compute_type
                )
                logger.info("Whisper model loaded successfully")
    return _model


def _materialize(content: bytes, extension: str) -> str:
    fd, path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise
    return path


def _extract_audio_from_video(video_path: str) -> str:
    fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path, "-y"],
        check=True, capture_output=True,
    )
    return audio_path


@execution_handler("transcribe")
def transcribe(payload) -> dict:
    ext = payload.get("extension")
    if not isinstance(ext, str) or ext.lower() not in (
        _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS
    ):
        raise ValueError("Transcribe requires a supported media extension")
    ext = ext.lower()
    content = (payload.get("_input_artifacts") or {}).get("media")
    if content is None:
        raise ValueError("Transcribe step is missing its media artifact")

    source_path = _materialize(content, ext)
    is_video = ext.lower() in _VIDEO_EXTENSIONS
    audio_path = source_path
    temp_audio = None

    if is_video:
        logger.info("Extracting audio from video: %s", source_path)
        temp_audio = _extract_audio_from_video(source_path)
        audio_path = temp_audio

    try:
        config = get_task_config("transcribe")
        beam_size = config.get("beam_size", 5)

        model = _get_model()
        with _inference_lock:
            segments, info = model.transcribe(
                audio_path, beam_size=beam_size
            )
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())

        transcript = " ".join(transcript_parts)

        logger.info(
            "Transcription complete: language=%s (prob=%.2f), duration=%.1fs, chars=%d",
            info.language, info.language_probability, info.duration, len(transcript),
        )

        return {
            "transcript": transcript,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
    finally:
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if os.path.exists(source_path):
            os.remove(source_path)
