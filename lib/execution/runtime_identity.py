import hashlib
import json
import platform
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[2]


def runtime_fingerprint(project_root: Path | None = None) -> str:
    root = project_root or _PROJECT_DIR
    lock = root / "requirements.txt"
    identity = {
        "kind": "python",
        "implementation": sys.implementation.name,
        "python": platform.python_version(),
        "platform": sys.platform,
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "requirementsSha256": (
            hashlib.sha256(lock.read_bytes()).hexdigest()
            if lock.is_file() and not lock.is_symlink()
            else None
        ),
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
