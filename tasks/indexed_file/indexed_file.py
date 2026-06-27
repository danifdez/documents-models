"""Job handlers for the assistant working-folder table.

Separate from `tasks/ingest/ingest.py` because the storage scope is different
(`indexed_file_chunks` vs. workspace `rag_chunks`) and we want the two paths to
be unable to leak into each other at the source. They share the same embedding
service and pgvector infrastructure underneath.
"""

import logging
import uuid
from typing import List

from services.text import semantic_chunk_text
from utils.job_registry import job_handler
from database.rag import get_folder_rag, PointStruct
from services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def _source_id(indexed_file_id: int) -> str:
    return f"indexed_file_{int(indexed_file_id)}"


def _owner_tag(owner_type: str, owner_id: int) -> str:
    """Stable tag used in the vector payload + as the indexed owner filter key."""
    return f"{owner_type}:{int(owner_id)}"


@job_handler("indexed-file-ingest")
def ingest_indexed_file(payload: dict) -> dict:
    """Vectorize the extracted text of an IndexedFile and upsert into the
    folder collection. Idempotent: deletes prior points for this `source_id`
    before upserting.

    Payload keys:
        - indexedFileId (int): IndexedFile id.
        - ownerType (str): 'main-assistant' or 'agent'.
        - ownerId (int): owning entity id.
        - content (str): extracted text already produced by T07.
        - filename (str): filename relative to the working folder.
        - checksum (str): content checksum, echoed back so the backend can
          ignore stale results if the file changed again in the meantime.
    """
    indexed_file_id = int(payload["indexedFileId"])
    owner_type = str(payload.get("ownerType") or "main-assistant")
    owner_id = int(payload.get("ownerId") or payload.get("assistantId") or 0)
    content = (payload.get("content") or "").strip()
    filename = payload.get("filename") or ""
    checksum = payload.get("checksum") or ""

    rag = get_folder_rag()
    source_id = _source_id(indexed_file_id)

    rag.delete_by_column("indexed_file_id", indexed_file_id)

    if not content:
        return {
            "success": True,
            "indexedFileId": indexed_file_id,
            "sourceId": source_id,
            "chunks": 0,
            "checksum": checksum,
        }

    chunks = semantic_chunk_text(content)
    embeddings = get_embedding_service().encode(chunks, normalize_embeddings=True)

    owner_tag = _owner_tag(owner_type, owner_id)
    points: List[PointStruct] = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "source_id": source_id,
                # Owner isolation is enforced by the indexed `owner_tag` column.
                "owner_tag": owner_tag,
                "owner_type": owner_type,
                "owner_id": int(owner_id),
                "indexed_file_id": indexed_file_id,
                "filename": filename,
                "part_number": i,
                "total_chunks": len(chunks),
            },
        ))

    if points:
        rag.upsert_points(points)

    return {
        "success": True,
        "indexedFileId": indexed_file_id,
        "sourceId": source_id,
        "chunks": len(points),
        "checksum": checksum,
    }


@job_handler("indexed-file-search")
def search_indexed_files(payload: dict) -> dict:
    """Semantic search over an owner's folder files. Returns hits with
    `indexedFileId`, `filename`, `snippet`, `score`. Hits from the same file
    are aggregated to a single entry (best snippet kept).

    Payload keys:
        - ownerType (str): 'main-assistant' or 'agent'.
        - ownerId (int): owning entity id.
        - query (str): user query.
        - limit (int): max results (default 10).
        - score_threshold (float, optional): minimum cosine score.
    """
    owner_type = str(payload.get("ownerType") or "main-assistant")
    owner_id = int(payload.get("ownerId") or payload.get("assistantId") or 0)
    query = (payload.get("query") or "").strip()
    limit = int(payload.get("limit") or 10)
    score_threshold = payload.get("score_threshold")

    if not query:
        return {"results": []}

    rag = get_folder_rag()
    query_embedding = get_embedding_service().encode_query(query)

    # Strict owner isolation via the indexed `owner_tag` column — a
    # `main-assistant` and an `agent` that share an id never leak into each
    # other's results.
    owner_tag = _owner_tag(owner_type, owner_id)
    points = rag.query_points(
        query_embedding,
        limit=max(limit * 3, limit),
        with_payload=True,
        owner_tag=owner_tag,
        score_threshold=score_threshold,
    )

    aggregated: dict = {}
    for p in points:
        payload_p = getattr(p, "payload", {}) or {}
        file_id = payload_p.get("indexed_file_id")
        if file_id is None:
            continue
        score = float(getattr(p, "score", 0.0))
        snippet = (payload_p.get("text") or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        existing = aggregated.get(file_id)
        if existing is None or score > existing["score"]:
            aggregated[file_id] = {
                "indexedFileId": file_id,
                "filename": payload_p.get("filename") or "",
                "snippet": snippet,
                "score": score,
            }

    results = sorted(aggregated.values(), key=lambda r: r["score"], reverse=True)[:limit]
    return {"results": results}


@job_handler("indexed-file-delete-vectors")
def delete_indexed_file_vectors(payload: dict) -> dict:
    """Delete vectors for a specific IndexedFile, or for all files of an
    owner. Idempotent.

    The per-file mode is mostly redundant now (the FK to ``indexed_files``
    cascades on delete), but kept for manual/idempotent cleanup. The owner-scoped
    wipe has no single parent row, so it stays job-driven.

    Payload keys (exactly one):
        - sourceId (str): e.g. `indexed_file_42`. Deletes that file only.
        - indexedFileId (int): convenience equivalent of `sourceId`.
        - ownerType + ownerId: wipes all of that owner's vectors.
    """
    rag = get_folder_rag()

    indexed_file_id = payload.get("indexedFileId")
    if indexed_file_id is None and payload.get("sourceId"):
        # Parse the trailing id out of `indexed_file_<id>`.
        try:
            indexed_file_id = int(str(payload["sourceId"]).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return {"error": "invalid sourceId"}

    if indexed_file_id is not None:
        rag.delete_by_column("indexed_file_id", int(indexed_file_id))
        return {"success": True, "indexedFileId": int(indexed_file_id)}

    # Owner-scoped wipe: backward-compatible with the older `assistantId` key.
    owner_type = payload.get("ownerType")
    owner_id = payload.get("ownerId")
    if owner_id is None and payload.get("assistantId") is not None:
        owner_type = owner_type or "main-assistant"
        owner_id = payload["assistantId"]
    if owner_id is not None and owner_type:
        rag.delete_by_column("owner_tag", _owner_tag(owner_type, int(owner_id)))
        return {"success": True, "ownerType": owner_type, "ownerId": int(owner_id)}

    return {"error": "must provide sourceId, indexedFileId or ownerType+ownerId"}
