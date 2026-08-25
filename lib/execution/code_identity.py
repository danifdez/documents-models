import fnmatch
import hashlib
import json
import os
import stat
import unicodedata
from functools import lru_cache
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_HASH_PREFIX = "sha256:"


def code_fingerprint(project_root: Path | None = None) -> str:
    supplied = os.environ.get("DOCUMENTS_CODE_FINGERPRINT", "").strip()
    if supplied:
        if not _is_hash(supplied):
            raise ValueError("DOCUMENTS_CODE_FINGERPRINT must be a SHA-256 reference")
        return supplied
    return _tree_fingerprint((project_root or _PROJECT_DIR).resolve())


@lru_cache(maxsize=4)
def _tree_fingerprint(root: Path) -> str:
    manifest = _tree_manifest(root)
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _HASH_PREFIX + hashlib.sha256(encoded).hexdigest()


def _tree_manifest(root: Path) -> dict:
    identity = json.loads(
        (root / "common" / "code_identity.json").read_text(encoding="utf-8")
    )
    if not isinstance(identity, dict) or identity.get("schema") != 1:
        raise ValueError("unsupported code identity schema")
    scopes = _patterns(identity, "versionScope")
    excludes = _patterns(identity, "exclude")
    entries = []
    stack = [root]
    while stack:
        directory = stack.pop()
        for path in sorted(directory.iterdir(), reverse=True):
            relative = unicodedata.normalize(
                "NFC", path.relative_to(root).as_posix()
            )
            included = _matches_any(relative, scopes)
            excluded = _matches_any(relative, excludes)
            if path.is_symlink():
                if included or _may_contain(relative, scopes):
                    raise ValueError(
                        f"code identity scope contains a symlink: {relative}"
                    )
                continue
            if path.is_dir():
                if not excluded and _may_contain(relative, scopes):
                    stack.append(path)
                continue
            if not included or excluded or not path.is_file():
                continue
            metadata = path.stat()
            if metadata.st_nlink != 1:
                raise ValueError(
                    f"code identity scope contains a hardlink: {relative}"
                )
            body = path.read_bytes()
            mode = (
                0o755
                if metadata.st_mode
                & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                else 0o644
            )
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "size": len(body),
                    "mode": f"{mode:04o}",
                }
            )
    return {"schema": 1, "entries": sorted(entries, key=lambda item: item["path"])}


def _patterns(identity: dict, key: str) -> tuple[str, ...]:
    value = identity.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"code identity {key} must be a non-empty string list")
    return tuple(value)


def _matches(path: str, pattern: str) -> bool:
    if fnmatch.fnmatchcase(path, pattern):
        return True
    if pattern.startswith("**/"):
        return _matches(path, pattern[3:])
    if pattern.endswith("/**") and not any(
        character in pattern[:-3] for character in "*?["
    ):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return False


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches(path, pattern) for pattern in patterns)


def _may_contain(directory: str, scopes: tuple[str, ...]) -> bool:
    prefix = directory + "/"
    for scope in scopes:
        base = scope[:-3].rstrip("/") if scope.endswith("/**") else scope
        if (
            base == directory
            or base.startswith(prefix)
            or directory.startswith(base + "/")
        ):
            return True
    return False


def _is_hash(value: str) -> bool:
    return (
        len(value) == 71
        and value.startswith(_HASH_PREFIX)
        and all(character in "0123456789abcdef" for character in value[7:])
    )
