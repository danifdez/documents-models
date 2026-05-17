"""Job handlers for the assistant working-folder collection.

Separate from `tasks/ingest/ingest.py` because the storage scope is different
(folder collection vs. workspace `rag_docs`) and we want the two paths to be
unable to leak into each other at the source. They share the same embedding
service and Qdrant infrastructure underneath.
"""

import logging
import uuid
from typing import List

from qdrant_client.models import PointStruct
from services.text import semantic_chunk_text
from utils.job_registry import job_handler
from database.rag import get_folder_rag
from services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def _source_id(indexed_file_id: int) -> str:
    return f"indexed_file_{int(indexed_file_id)}"


@job_handler("indexed-file-ingest")
def ingest_indexed_file(payload: dict) -> dict:
    """Vectorize the extracted text of an IndexedFile and upsert into the
    folder collection. Idempotent: deletes prior points for this `source_id`
    before upserting.

    Payload keys:
        - indexedFileId (int): IndexedFile id.
        - assistantId (int): owning assistant.
        - content (str): extracted text already produced by T07.
        - filename (str): filename relative to the working folder.
        - checksum (str): content checksum, echoed back so the backend can
          ignore stale results if the file changed again in the meantime.
    """
    indexed_file_id = int(payload["indexedFileId"])
    assistant_id = int(payload["assistantId"])
    content = (payload.get("content") or "").strip()
    filename = payload.get("filename") or ""
    checksum = payload.get("checksum") or ""

    rag = get_folder_rag()
    source_id = _source_id(indexed_file_id)

    rag.delete_by_source(source_id)

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

    points: List[PointStruct] = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "source_id": source_id,
                "assistant_id": str(assistant_id),
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
    """Semantic search over an assistant's folder files. Returns hits with
    `indexedFileId`, `filename`, `snippet`, `score`. Hits from the same file
    are aggregated to a single entry (best snippet kept).

    Payload keys:
        - assistantId (int): scope filter.
        - query (str): user query.
        - limit (int): max results (default 10).
        - score_threshold (float, optional): minimum cosine score.
    """
    assistant_id = int(payload["assistantId"])
    query = (payload.get("query") or "").strip()
    limit = int(payload.get("limit") or 10)
    score_threshold = payload.get("score_threshold")

    if not query:
        return {"results": []}

    rag = get_folder_rag()
    query_embedding = get_embedding_service().encode_query(query)

    points = rag.query_points(
        query_embedding,
        limit=max(limit * 3, limit),
        with_payload=True,
        assistant_id=str(assistant_id),
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
    assistant. Idempotent.

    Payload keys (exactly one):
        - sourceId (str): e.g. `indexed_file_42`. Deletes that file only.
        - indexedFileId (int): convenience equivalent of `sourceId`.
        - assistantId (int): wipes all of the assistant's vectors.
    """
    rag = get_folder_rag()

    source_id = payload.get("sourceId")
    if not source_id and payload.get("indexedFileId") is not None:
        source_id = _source_id(int(payload["indexedFileId"]))

    if source_id:
        rag.delete_by_source(source_id)
        return {"success": True, "sourceId": source_id}

    assistant_id = payload.get("assistantId")
    if assistant_id is not None:
        rag.delete_by_assistant(str(assistant_id))
        return {"success": True, "assistantId": int(assistant_id)}

    return {"error": "must provide sourceId, indexedFileId or assistantId"}
