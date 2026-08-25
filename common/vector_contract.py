"""Pure helpers for self-contained vector assignments."""

import json
import math
from typing import Any, Dict, List

EMBEDDING_DIMENSIONS = 384
MAX_CANDIDATES = 5_000


def load_vector_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts = payload.get("_input_artifacts") or {}
    raw = artifacts.get("vector_candidates")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("vector_candidates artifact is required")
    try:
        document = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("vector_candidates artifact is invalid") from error
    candidates = document.get("candidates") if isinstance(document, dict) else None
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES:
        raise ValueError("vector candidates must be a bounded array")
    return [_validate_candidate(candidate) for candidate in candidates]


def rank_vector_candidates(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    limit: int,
    score_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    query = _validate_embedding(query_embedding)
    query_norm = math.sqrt(sum(value * value for value in query))
    if query_norm == 0:
        raise ValueError("query embedding cannot be zero")
    ranked = []
    for candidate in candidates:
        embedding = candidate["embedding"]
        candidate_norm = math.sqrt(sum(value * value for value in embedding))
        if candidate_norm == 0:
            continue
        score = sum(a * b for a, b in zip(query, embedding)) / (
            query_norm * candidate_norm
        )
        if score_threshold is not None and score < score_threshold:
            continue
        ranked.append({**candidate, "score": float(score)})
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return ranked[: max(1, min(int(limit), 100))]


def vector_point(
    point_id: str,
    embedding: Any,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(point_id, str) or not point_id.strip():
        raise ValueError("vector point id is required")
    if not isinstance(payload, dict):
        raise ValueError("vector point payload must be an object")
    return {
        "id": point_id.strip(),
        "embedding": _validate_embedding(embedding),
        "payload": payload,
    }


def _validate_candidate(candidate: Any) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("vector candidate must be an object")
    candidate_id = candidate.get("id")
    payload = candidate.get("payload")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("vector candidate id is required")
    if not isinstance(payload, dict):
        raise ValueError("vector candidate payload must be an object")
    return {
        "id": candidate_id.strip(),
        "embedding": _validate_embedding(candidate.get("embedding")),
        "payload": payload,
    }


def _validate_embedding(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"embedding must contain {EMBEDDING_DIMENSIONS} values"
        )
    embedding = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("embedding values must be numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite")
        embedding.append(number)
    return embedding
