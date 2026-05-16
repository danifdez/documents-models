"""Entry point script for the live dictation worker.

Run as:
    python -m voice.run_live_worker
    # or
    python voice/run_live_worker.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Allow direct execution as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voice.live_worker import main  # noqa: E402


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
