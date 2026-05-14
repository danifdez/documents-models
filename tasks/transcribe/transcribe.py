import os
import logging
import subprocess
import tempfile

from utils.job_registry import job_handler
from services.model_config import get_task_config

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.opus', '.aiff', '.aif'}
_VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm', '.wmv'}

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        config = get_task_config("transcribe")
        model_size = config.get("model", "base")
        device = config.get("device", "cpu")
        compute_type = config.get("compute_type", "int8")

        logger.info("Loading Whisper model: %s (device=%s, compute_type=%s)", model_size, device, compute_type)
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")
    return _model


def _materialize(input_blob: bytes, extension: str) -> str:
    fd, path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(input_blob)
    except Exception:
        os.close(fd)
        if os.path.exists(path):
            os.remove(path)
        raise
    return path


def _extract_audio_from_video(video_path: str) -> str:
    audio_path = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path, "-y"],
        check=True, capture_output=True,
    )
    return audio_path


@job_handler("transcribe")
def transcribe(payload) -> dict:
    ext = payload["extension"]
    input_blob = payload.get("_input_blob")
    if input_blob is None:
        return {"error": "transcribe job is missing input_blob"}

    source_path = _materialize(input_blob, ext)
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
        segments, info = model.transcribe(audio_path, beam_size=beam_size)

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
