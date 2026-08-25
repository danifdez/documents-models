"""Self-contained embedding and search steps for assistant memory."""

from common.execution_registry import execution_handler
from common.vector_contract import load_vector_candidates, rank_vector_candidates
from services.embedding_service import get_embedding_service


@execution_handler("memory-ingest")
def ingest_memory(payload: dict) -> dict:
    memory_id = int(payload["memoryId"])
    name = payload.get("name")
    body = payload.get("body")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("memory-ingest name is required")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("memory-ingest body is required")
    embedding = get_embedding_service().encode_single(
        f"{name.strip()}: {body.strip()}"
    )
    return {"memoryId": memory_id, "embedding": embedding.tolist()}


@execution_handler("memory-search")
def search_memory(payload: dict) -> dict:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"results": []}
    limit = max(1, min(int(payload.get("limit") or 8), 32))
    candidates = load_vector_candidates(payload)
    query_embedding = get_embedding_service().encode_query(query.strip()).tolist()
    ranked = rank_vector_candidates(query_embedding, candidates, limit)
    results = []
    for hit in ranked:
        metadata = hit["payload"]
        memory_id = metadata.get("memory_id")
        if memory_id is None:
            continue
        results.append(
            {
                "memoryId": int(memory_id),
                "score": hit["score"],
                "name": str(metadata.get("name") or ""),
                "type": str(metadata.get("type") or "fact"),
            }
        )
    return {"results": results}
