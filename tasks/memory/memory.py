"""Job handlers for the assistant personal-memory table.

Separate from `tasks/ingest/ingest.py` and `tasks/indexed_file/indexed_file.py`
because the storage scope is different (assistant memory entries in their own
`memory_vectors` table, 1-to-1 with `assistant_memory_entries` via FK). The
embedding model is the shared, multilingual `EmbeddingService` — same as every
other table.
"""

import logging

import psycopg
from psycopg.rows import dict_row

from utils.job_registry import job_handler
from database.rag import get_memory_rag, PointStruct
from services.embedding_service import get_embedding_service
from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)

logger = logging.getLogger(__name__)


@job_handler("memory-ingest")
def ingest_memory(payload: dict) -> dict:
    """Embed ``name + ": " + body`` and upsert into ``memory_vectors``.

    Idempotent: the point id ``memory_<memoryId>`` is stable, so a re-ingest
    of the same memory overwrites the prior vector. Used both after
    create and after update from the backend.

    Payload keys:
        - memoryId (int): required.
        - assistantId (int): required, stored as filter key.
        - name (str): required, concatenated for the embedding.
        - type (str): 'fact' | 'event' | 'instruction'.
        - body (str): required, concatenated for the embedding.
    """
    try:
        memory_id = int(payload["memoryId"])
        assistant_id = int(payload["assistantId"])
        name = (payload.get("name") or "").strip()
        type_ = (payload.get("type") or "fact").strip().lower()
        body = (payload.get("body") or "").strip()
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("memory-ingest: invalid payload: %s", e)
        return {"error": "invalid payload"}

    if not name or not body:
        logger.info("memory-ingest: skip empty memory %s", memory_id)
        return {"success": True, "memoryId": memory_id, "skipped": "empty"}

    text = f"{name}: {body}"
    embedding = get_embedding_service().encode_single(text)
    # PK is `memory_id` (taken from payload) so a re-ingest upserts in place.
    point = PointStruct(
        id=memory_id,
        vector=embedding.tolist(),
        payload={
            "memory_id": memory_id,
            "assistant_id": str(assistant_id),
            "name": name,
            "type": type_,
        },
    )
    get_memory_rag().upsert_points([point])
    return {"success": True, "memoryId": memory_id}


@job_handler("memory-search")
def search_memory(payload: dict) -> dict:
    """Semantic search over ``memory_vectors`` filtered by assistant_id.

    Payload keys:
        - assistantId (int): required, filter key.
        - query (str): required, natural-language query.
        - limit (int, optional): top-K, default 8.

    Returns:
        ``{ "results": [{ "memoryId": int, "score": float, "name": str, "type": str }, ...] }``

        The body is NOT included — the caller looks it up in the SQL DB
        by memoryId. The payload also returns ``name`` and ``type`` for
        convenience in case the caller wants to render hints without a
        second DB hit.
    """
    try:
        assistant_id = int(payload["assistantId"])
        query = (payload.get("query") or "").strip()
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("memory-search: invalid payload: %s", e)
        return {"error": "invalid payload"}

    if not query:
        return {"results": []}

    limit = int(payload.get("limit") or 8)
    limit = max(1, min(limit, 32))  # clamp

    embedding = get_embedding_service().encode_query(query)
    hits = get_memory_rag().query_points(
        query_vector=embedding.tolist(),
        limit=limit,
        assistant_id=str(assistant_id),
        with_payload=True,
    )

    results = []
    for h in hits:
        pl = h.payload or {}
        mid = pl.get("memory_id")
        if mid is None:
            continue
        results.append({
            "memoryId": int(mid),
            "score": float(h.score) if h.score is not None else 0.0,
            "name": pl.get("name") or "",
            "type": pl.get("type") or "fact",
        })
    return {"results": results}


def _memory_db_connection():
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        autocommit=True,
        row_factory=dict_row,
    )


def _recent_entries(assistant_id: int, limit: int) -> list:
    with _memory_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, type, body FROM assistant_memory_entries "
            "WHERE assistant_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (assistant_id, limit),
        )
        return cur.fetchall()


def _entries_by_ids(assistant_id: int, ids: list) -> dict:
    if not ids:
        return {}
    with _memory_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, type, body FROM assistant_memory_entries "
            "WHERE assistant_id = %s AND id = ANY(%s)",
            (assistant_id, list(ids)),
        )
        return {r["id"]: r for r in cur.fetchall()}


def relevant_for_injection(assistant_id: int, query: str, limit: int = 8) -> list:
    """Memories likely relevant to the user's message, mixed with recent ones.
    In-process replacement for the backend's old relevantForInjection; returns
    [{id, name, type, body, relevance}] with relevance in high|medium|recent.
    """
    clean = (query or "").strip()
    recent_limit, cap = 5, 12
    if len(clean) < 8:
        return [
            {**e, "relevance": "recent"}
            for e in _recent_entries(assistant_id, recent_limit)
        ]
    try:
        hits = search_memory(
            {"assistantId": assistant_id, "query": clean, "limit": limit}
        ).get("results", [])
    except Exception:
        logger.exception("memory: relevant_for_injection search failed")
        hits = []
    bodies = _entries_by_ids(assistant_id, [h["memoryId"] for h in hits])
    seen: set = set()
    merged: list = []
    for h in hits:
        e = bodies.get(h["memoryId"])
        if not e:
            continue
        e["relevance"] = "high" if (h.get("score") or 0) > 0.85 else "medium"
        merged.append(e)
        seen.add(e["id"])
    for r in _recent_entries(assistant_id, recent_limit):
        if r["id"] in seen:
            continue
        r["relevance"] = "recent"
        merged.append(r)
        seen.add(r["id"])
        if len(merged) >= cap:
            break
    return merged[:cap]


@job_handler("memory-delete-vectors")
def delete_memory_vectors(payload: dict) -> dict:
    """Delete memory vectors.

    Two modes:
    - Single: payload has ``memoryId`` → delete the row with that PK. Mostly
      redundant now (the FK to ``assistant_memory_entries`` cascades on delete),
      kept for manual/idempotent cleanup.
    - Bulk: payload has ``assistantId`` (and no ``memoryId``) → delete all rows
      where ``assistant_id == <assistantId>``. Used after
      ``AssistantMemoryService.clear``.

    Idempotent: deleting a row that does not exist is a no-op.
    """
    memory_id = payload.get("memoryId")
    assistant_id = payload.get("assistantId")
    rag = get_memory_rag()
    if memory_id is not None:
        try:
            mid = int(memory_id)
        except (TypeError, ValueError):
            return {"error": "invalid memoryId"}
        rag.delete_points([mid])
        return {"success": True, "deleted": "single", "memoryId": mid}
    if assistant_id is not None:
        try:
            aid = int(assistant_id)
        except (TypeError, ValueError):
            return {"error": "invalid assistantId"}
        rag.delete_by_column("assistant_id", aid)
        return {"success": True, "deleted": "bulk", "assistantId": aid}
    return {"error": "memoryId or assistantId required"}
