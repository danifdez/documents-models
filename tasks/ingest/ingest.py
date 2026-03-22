import uuid
from services.text import semantic_chunk_text, clean_html_text
from utils.job_registry import job_handler
from qdrant_client.models import PointStruct
from database.rag import get_rag
from services.embedding_service import get_embedding_service


@job_handler("ingest-content")
def ingest(payload) -> dict:
    """
    Ingests content for a resource or doc into the vector database.
    Deletes old vectors before upserting new ones to keep data in sync.

    Payload keys:
        - content (str): HTML content to ingest
        - sourceType (str): "resource" or "doc" (default: "resource")
        - resourceId (int): Required when sourceType is "resource"
        - docId (int): Required when sourceType is "doc"
        - projectId (int): Project ID for filtering
    """
    database = get_rag()
    embedding_service = get_embedding_service()

    source_type = payload.get("sourceType", "resource")
    if source_type == "doc":
        source_id = f"doc_{payload['docId']}"
    elif source_type == "knowledge":
        source_id = f"knowledge_{payload['knowledgeEntryId']}"
    else:
        source_id = str(payload["resourceId"])

    project_id = str(payload["projectId"]) if payload.get("projectId") else None

    # Always delete old vectors for this source before re-ingesting
    database.delete_by_source(source_id)

    clean_content = clean_html_text(payload["content"])

    if not clean_content:
        return {"success": True}

    chunks = semantic_chunk_text(clean_content)
    chunk_embeddings = embedding_service.encode(chunks, normalize_embeddings=True)

    total_chunks = len(chunks)
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings), 1):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source_id": source_id,
                    "project_id": project_id,
                    "source_type": source_type,
                    "part_number": i,
                    "total_chunks": total_chunks,
                }
            )
        )

    if points:
        database.upsert_points(points=points)

    return {"success": True}


@job_handler("delete-vectors")
def delete_vectors(payload) -> dict:
    """Delete all vectors for a given source_id from the vector database."""
    source_id = payload.get("sourceId")
    if not source_id:
        return {"error": "sourceId is required"}

    database = get_rag()
    database.delete_by_source(source_id)
    return {"success": True}
