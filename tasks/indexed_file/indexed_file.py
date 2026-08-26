"""Self-contained embedding and search steps for indexed files."""

from common.execution_registry import execution_handler
from common.vector_contract import (
    load_vector_candidates,
    rank_vector_candidates,
    vector_point,
    vector_points_output,
)
from lib.execution.output_artifact import HandlerOutput
from services.embedding_service import get_embedding_service
from services.text import semantic_chunk_text


@execution_handler("indexed-file-ingest")
def ingest_indexed_file(payload: dict) -> HandlerOutput:
    indexed_file_id = int(payload["indexedFileId"])
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("indexed-file-ingest content must be a string")
    content = content.strip()
    filename = str(payload.get("filename") or "")
    checksum = str(payload.get("checksum") or "")
    source_id = f"indexed_file_{indexed_file_id}"
    if not content:
        return vector_points_output(
            {
                "indexedFileId": indexed_file_id,
                "sourceId": source_id,
                "chunks": 0,
                "checksum": checksum,
            },
            [],
        )

    chunks = semantic_chunk_text(content)
    embeddings = get_embedding_service().encode(
        chunks, normalize_embeddings=True
    )
    points = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
        points.append(
            vector_point(
                f"{source_id}:{index}",
                embedding.tolist(),
                {
                    "text": chunk,
                    "source_id": source_id,
                    "indexed_file_id": indexed_file_id,
                    "filename": filename,
                    "part_number": index,
                    "total_chunks": len(chunks),
                },
            )
        )
    return vector_points_output(
        {
            "indexedFileId": indexed_file_id,
            "sourceId": source_id,
            "chunks": len(points),
            "checksum": checksum,
        },
        points,
    )


@execution_handler("indexed-file-search")
def search_indexed_files(payload: dict) -> dict:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"results": []}
    limit = max(1, min(int(payload.get("limit") or 10), 100))
    threshold = payload.get("score_threshold")
    if threshold is not None:
        threshold = float(threshold)
    candidates = load_vector_candidates(payload)
    query_embedding = get_embedding_service().encode_query(query.strip()).tolist()
    ranked = rank_vector_candidates(
        query_embedding,
        candidates,
        max(limit * 3, limit),
        threshold,
    )
    aggregated = {}
    for hit in ranked:
        metadata = hit["payload"]
        file_id = metadata.get("indexed_file_id")
        if file_id is None:
            continue
        score = hit["score"]
        snippet = str(metadata.get("text") or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        existing = aggregated.get(file_id)
        if existing is None or score > existing["score"]:
            aggregated[file_id] = {
                "indexedFileId": int(file_id),
                "filename": str(metadata.get("filename") or ""),
                "snippet": snippet,
                "score": score,
            }
    results = sorted(
        aggregated.values(),
        key=lambda item: (-item["score"], item["indexedFileId"]),
    )[:limit]
    return {"results": results}
