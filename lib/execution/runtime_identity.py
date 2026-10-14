import hashlib
import os
import platform
import sys
from pathlib import Path

from common.canonical_json import canonical_sha256, is_sha256_reference

_PROJECT_DIR = Path(__file__).resolve().parents[2]


def runtime_fingerprint(project_root: Path | None = None) -> str:
    supplied = os.environ.get("DOCUMENTS_RUNTIME_FINGERPRINT", "").strip()
    if supplied:
        if not is_sha256_reference(supplied):
            raise ValueError(
                "DOCUMENTS_RUNTIME_FINGERPRINT must be a SHA-256 reference"
            )
        return supplied
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
    return canonical_sha256(identity)
