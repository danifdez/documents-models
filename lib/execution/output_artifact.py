"""Canonical handler outputs that carry durable attempt artifacts."""

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

MAX_OUTPUT_ARTIFACT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class OutputArtifact:
    role: str
    kind: str
    body: bytes


@dataclass(frozen=True)
class HandlerOutput:
    value: Any
    artifacts: List[OutputArtifact]


def json_output_artifact(role: str, kind: str, value: Any) -> OutputArtifact:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(body) > MAX_OUTPUT_ARTIFACT_BYTES:
        raise ValueError(f"{kind} output artifact exceeds 8 MiB")
    return OutputArtifact(role=role, kind=kind, body=body)


def prepare_output_artifacts(
    artifacts: List[OutputArtifact],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    uploads = []
    refs = []
    revisions: Dict[str, int] = {}
    for artifact in artifacts:
        if not artifact.role.strip() or not artifact.kind.strip():
            raise ValueError("Output artifact role and kind are required")
        if len(artifact.body) > MAX_OUTPUT_ARTIFACT_BYTES:
            raise ValueError(f"{artifact.kind} output artifact exceeds 8 MiB")
        artifact_id = str(uuid.uuid4())
        revisions[artifact.role] = revisions.get(artifact.role, 0) + 1
        uploads.append(
            {
                "artifactId": artifact_id,
                "kind": artifact.kind,
                "contentHash": "sha256:"
                + hashlib.sha256(artifact.body).hexdigest(),
                "size": len(artifact.body),
                "mediaType": "application/json",
                "bodyBase64": base64.b64encode(artifact.body).decode("ascii"),
            }
        )
        refs.append(
            {
                "role": artifact.role,
                "artifactId": artifact_id,
                "revision": revisions[artifact.role],
            }
        )
    return uploads, refs
