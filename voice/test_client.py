"""Minimal QA client for the live dictation worker.

Usage:
    python voice/test_client.py path/to/sample.wav [--chunk-ms 500]

Streams the audio file in chunks to ws://localhost:9100/voice and prints
incoming partials. The file is converted to webm/opus via ffmpeg on the fly so
it matches what the browser would send.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile

# Ensure project root is importable when run as script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger("voice.test_client")


def _encode_to_webm(input_path: str) -> str:
    out_path = tempfile.mktemp(suffix=".webm")
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", input_path,
            "-vn",
            "-c:a", "libopus", "-b:a", "32k", "-ar", "16000", "-ac", "1",
            out_path,
        ],
        check=True,
    )
    return out_path


async def stream_file(uri: str, audio_path: str, chunk_ms: int) -> None:
    import websockets  # type: ignore

    encoded = _encode_to_webm(audio_path)
    try:
        with open(encoded, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(encoded)
        except OSError:
            pass

    # Heuristic: 32 kbps opus -> 4 KB/s -> 2 KB per 500 ms.
    chunk_size = max(1, (32 * 1024 // 8) * chunk_ms // 1000)
    logger.info("Streaming %d bytes in chunks of %d", len(data), chunk_size)

    async with websockets.connect(uri) as ws:
        async def reader():
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    print(f"[raw] {msg}")
                    continue
                if payload.get("type") == "partial":
                    tag = "FINAL" if payload.get("isFinal") else "    "
                    print(f"[{tag}] {payload.get('text')}")
                elif payload.get("type") == "error":
                    print(f"[ERROR] {payload.get('message')}")

        reader_task = asyncio.create_task(reader())

        for i in range(0, len(data), chunk_size):
            await ws.send(data[i:i + chunk_size])
            await asyncio.sleep(chunk_ms / 1000)

        await ws.send(json.dumps({"type": "stop"}))
        await reader_task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--uri", default="ws://localhost:9100/voice")
    parser.add_argument("--chunk-ms", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(stream_file(args.uri, args.audio, args.chunk_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
