"""
Live dictation worker.

Persistent process that keeps a `faster-whisper` model in memory and processes
audio chunks streamed over a WebSocket. Sits outside the normal job queue
(`jobs.py`) on purpose — live dictation cannot wait behind long-running jobs.

Protocol (`ws://<host>:<port>/voice`):
    Inbound from client:
        - binary frame  → audio chunk (webm/opus or wav). Appended to buffer.
        - text  {"type":"stop"}    → flush buffer and emit final partial.
        - text  {"type":"cancel"}  → drop buffer, close session, emit nothing.
    Outbound to client:
        - {"type":"partial", "text": str, "isFinal": bool}
        - {"type":"error",   "message": str}

The worker is intentionally agnostic about who its client is: the NestJS
`VoiceGateway` (Task 05) sits in front and reuses this same protocol.
"""
from __future__ import annotations

import asyncio
import http
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

# Allow running as `python live_worker.py` from the models dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.model_config import get_config, get_task_config  # noqa: E402

logger = logging.getLogger(__name__)

# Whisper model singleton (one model per process — Task 07 spawns N processes).
_model = None


def _live_config() -> dict:
    """Return live-dictation config, falling back to transcribe defaults."""
    base = {
        "model": "base",
        "device": "cpu",
        "compute_type": "int8",
        "window_seconds": 5,
        "overlap_seconds": 1,
        "vad_filter": True,
        "host": "127.0.0.1",
        "port": 9100,
    }
    base.update(get_task_config("transcribe") or {})
    base.update(get_config().get("voice_live", {}))
    return base


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        cfg = _live_config()
        logger.info(
            "Loading Whisper live model: %s (device=%s, compute=%s)",
            cfg["model"], cfg["device"], cfg["compute_type"],
        )
        _model = WhisperModel(
            cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"],
        )
        logger.info("Whisper live model loaded")
    return _model


@dataclass
class Session:
    """Per-connection state. One model, one buffer, one accumulated text."""
    sid: str
    buffer: bytearray = field(default_factory=bytearray)
    stable_text: str = ""
    last_window_text: str = ""
    bytes_since_last_run: int = 0

    @property
    def accumulated_text(self) -> str:
        out = self.stable_text
        if self.last_window_text:
            out = (out + " " + self.last_window_text).strip()
        return out


def _decode_to_wav(blob: bytes) -> Optional[str]:
    """Decode arbitrary container (webm/opus/ogg/wav) to 16 kHz mono PCM wav.

    Returns the path to a temp wav file, or None if ffmpeg fails. The caller
    is responsible for deleting the file.
    """
    in_fd, in_path = tempfile.mkstemp(suffix=".bin")
    out_path = tempfile.mktemp(suffix=".wav")
    try:
        with os.fdopen(in_fd, "wb") as f:
            f.write(blob)
        result = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", in_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                out_path,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            logger.warning("ffmpeg decode failed: %s", result.stderr.decode(errors="replace")[:200])
            return None
        return out_path
    finally:
        if os.path.exists(in_path):
            try:
                os.remove(in_path)
            except OSError:
                pass


def _transcribe_window(audio_path: str, vad_filter: bool) -> str:
    """Run Whisper on a wav file. Returns the concatenated text (auto-detected language)."""
    model = get_model()
    segments, _info = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
    )
    parts = [seg.text.strip() for seg in segments if seg.text]
    return " ".join(p for p in parts if p)


async def _run_inference(session: Session, *, is_final: bool) -> Optional[str]:
    """Decode the buffered audio and run Whisper. Returns the new accumulated
    text, or None if nothing usable came out. Runs the blocking call in a
    thread so the websocket loop stays responsive.
    """
    if not session.buffer:
        return session.accumulated_text if is_final else None

    blob = bytes(session.buffer)
    wav_path = await asyncio.to_thread(_decode_to_wav, blob)
    if wav_path is None:
        return None

    cfg = _live_config()
    try:
        text = await asyncio.to_thread(
            _transcribe_window, wav_path, cfg["vad_filter"],
        )
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    text = (text or "").strip()
    if is_final:
        # Last run — fold whatever we have into stable_text and emit.
        session.stable_text = (session.stable_text + " " + text).strip() if text else session.stable_text
        session.last_window_text = ""
    else:
        session.last_window_text = text
    return session.accumulated_text


# ── WebSocket server ────────────────────────────────────────────────────────

async def _send_json(ws, payload: dict) -> None:
    await ws.send(json.dumps(payload, ensure_ascii=False))


async def handle_connection(ws):
    """Process a single client connection (websockets >= 12 handler shape)."""
    sid = f"sess-{id(ws):x}"
    session = Session(sid=sid)
    cfg = _live_config()
    window_bytes_threshold = cfg["window_seconds"] * 16000 * 2  # rough heuristic; we re-decode the buffer

    logger.info("[%s] connected", sid)
    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                session.buffer.extend(msg)
                session.bytes_since_last_run += len(msg)
                if session.bytes_since_last_run >= window_bytes_threshold // 4:
                    session.bytes_since_last_run = 0
                    text = await _run_inference(session, is_final=False)
                    if text is not None:
                        await _send_json(ws, {"type": "partial", "text": text, "isFinal": False})
                continue

            # Text frame: control message.
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await _send_json(ws, {"type": "error", "message": "invalid control message"})
                continue

            kind = data.get("type")
            if kind == "stop":
                text = await _run_inference(session, is_final=True)
                await _send_json(ws, {
                    "type": "partial",
                    "text": text or session.accumulated_text,
                    "isFinal": True,
                })
                break
            if kind == "cancel":
                session.buffer.clear()
                break
            await _send_json(ws, {"type": "error", "message": f"unknown control {kind}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] worker error", sid)
        try:
            await _send_json(ws, {"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        logger.info("[%s] disconnected (buffer=%d bytes)", sid, len(session.buffer))


def _build_process_request():
    """Return a `process_request` callback that answers GET /health.

    `websockets.serve` invokes this before the upgrade. Returning a Response
    short-circuits the upgrade and answers as plain HTTP; returning None lets
    the WS handshake proceed. Signature targets the `websockets` 12+ API:
    `process_request(connection, request) -> Response | None`.
    """
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def process_request(_connection, request):
        if request.path == "/health":
            body = json.dumps({"status": "ok", "model_loaded": _model is not None}).encode()
            headers = Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ])
            return Response(
                status_code=http.HTTPStatus.OK,
                reason_phrase="OK",
                headers=headers,
                body=body,
            )
        return None

    return process_request


async def main():
    import websockets

    cfg = _live_config()
    host = cfg["host"]
    port = int(os.environ.get("VOICE_LIVE_PORT", cfg["port"]))

    # Eagerly load the model so the first connection is fast.
    get_model()

    logger.info("Voice live worker listening on ws://%s:%d/voice", host, port)
    async with websockets.serve(
        handle_connection,
        host,
        port,
        max_size=2 * 1024 * 1024,
        process_request=_build_process_request(),
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
