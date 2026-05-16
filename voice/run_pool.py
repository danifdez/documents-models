"""Spawn N `run_live_worker.py` processes on consecutive ports.

Usage:
    python -m voice.run_pool                  # N comes from VOICE_WORKER_POOL_SIZE (default 1)
    python -m voice.run_pool --workers 4
    python -m voice.run_pool --workers 4 --base-port 9100

Each worker is an independent process (not a thread) to isolate failures and
take advantage of SIMD/AVX per process. The supervisor forwards SIGTERM/SIGINT
to its children and waits for them to terminate.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Optional

logger = logging.getLogger("voice.pool")


def _spawn_worker(port: int, env_extra: Optional[dict] = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["VOICE_LIVE_PORT"] = str(port)
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, "-m", "voice.run_live_worker"]
    logger.info("Spawning worker on port %d (pid will follow)", port)
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--workers",
        type=int,
        default=int(os.environ.get("VOICE_WORKER_POOL_SIZE", 1)),
        help="Number of workers to launch (default: VOICE_WORKER_POOL_SIZE or 1)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=int(os.environ.get("VOICE_WORKER_BASE_PORT", 9100)),
        help="Port for the first worker; the rest use base+1, base+2, ...",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    procs: list[subprocess.Popen] = []
    ports = [args.base_port + i for i in range(args.workers)]
    logger.info("Pool size=%d ports=%s", args.workers, ports)

    for port in ports:
        procs.append(_spawn_worker(port))

    # Forward signals to children.
    stop_requested = False

    def _forward(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        logger.info("Signal %s received; terminating workers", signum)
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signum)
                except Exception:  # noqa: BLE001
                    pass

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    # Supervision loop: if a worker dies, we log it and continue.
    # No automatic respawn (Task 07 leaves it as an optional improvement).
    try:
        while procs and not stop_requested:
            time.sleep(1)
            for i, p in enumerate(procs):
                rc = p.poll()
                if rc is not None:
                    logger.warning("Worker on port %d exited with code %d", ports[i], rc)
            procs = [p for p, port in zip(procs, ports) if p.poll() is None]
    finally:
        # Orderly wait up to 5s and then kill.
        deadline = time.time() + 5
        for p in procs:
            remaining = max(0.1, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Worker pid=%d did not exit; killing", p.pid)
                p.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
