"""Canonical JSON encoding for content hashes and durable identities.

Single source of truth: the bytes produced here are hashed and compared outside
this process (artifact content hashes, frozen product-skill hashes, code and
runtime fingerprints), so the encoding is a contract — UTF-8, sorted keys,
compact separators, no NaN/Infinity. A site that hashes a JSON structure with
its own `json.dumps` call can make the same value hash differently.
"""

import hashlib
import json
from typing import Any

HASH_PREFIX = "sha256:"


def canonical_bytes(value: Any) -> bytes:
    """Encode a JSON-able value the way this project hashes it."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Reference hash of a JSON-able value, as `sha256:<64 lowercase hex>`."""
    return HASH_PREFIX + hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256_reference(value: str) -> bool:
    """True when `value` is exactly a reference produced by `canonical_sha256`."""
    digest = value[len(HASH_PREFIX):]
    return (
        len(value) == len(HASH_PREFIX) + 64
        and value.startswith(HASH_PREFIX)
        and all(character in "0123456789abcdef" for character in digest)
    )
