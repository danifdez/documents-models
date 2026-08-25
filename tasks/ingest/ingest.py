"""Self-contained workspace vectorization step."""

from common.execution_registry import execution_handler
from common.vector_contract import vector_point
from services.embedding_service import get_embedding_service
from services.text import clean_html_text, semantic_chunk_text


def _source(payload: dict) -> tuple[str, str]:
    source_type = str(payload.get("sourceType") or "resource")
    if source_type == "resource":
        return source_type, f"resource_{int(payload['resourceId'])}"
    if source_type == "doc":
        return source_type, f"doc_{int(payload['docId'])}"
    if source_type == "knowledge":
        return source_type, f"knowledge_{int(payload['knowledgeEntryId'])}"
    raise ValueError(f"Unsupported source type: {source_type}")


@execution_handler("ingest-content")
def ingest(payload: dict) -> dict:
    source_type, source_id = _source(payload)
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("ingest-content content must be a string")
    clean_content = clean_html_text(content)
    if not clean_content:
        return {"sourceId": source_id, "points": [], "chunks": 0}

    chunks = semantic_chunk_text(clean_content)
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
                    "project_id": payload.get("projectId"),
                    "source_type": source_type,
                    "part_number": index,
                    "total_chunks": len(chunks),
                },
            )
        )
    return {"sourceId": source_id, "points": points, "chunks": len(points)}
