"""Self-contained workspace vectorization step."""

from common.execution_registry import execution_handler
from common.vector_contract import (
    vector_points_for_chunks,
    vector_points_output,
)
from lib.execution.output_artifact import HandlerOutput
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
def ingest(payload: dict) -> HandlerOutput:
    source_type, source_id = _source(payload)
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("ingest-content content must be a string")
    clean_content = clean_html_text(content)
    if not clean_content:
        return vector_points_output(
            {"sourceId": source_id, "chunks": 0}, []
        )

    chunks = semantic_chunk_text(clean_content)
    embeddings = get_embedding_service().encode(
        chunks, normalize_embeddings=True
    )
    points = vector_points_for_chunks(
        source_id,
        chunks,
        embeddings,
        {
            "project_id": payload.get("projectId"),
            "source_type": source_type,
        },
    )
    return vector_points_output(
        {"sourceId": source_id, "chunks": len(points)}, points
    )
