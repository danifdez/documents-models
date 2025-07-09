import uuid
from services.text import chunk_text, clean_html_text
from utils.job_registry import job_handler
from qdrant_client.models import PointStruct
from database.rag import get_rag
from services.embedding_service import get_embedding_service

@job_handler("ingest-content")
def ingest(payload) -> dict:
    """
    Ingests the provided text for the given source and project.

    Args:
        source_id (str): The ID of the source.
        project_id (str): The ID of the project.
        content (str): The text to ingest.

    Returns:
        bool: True if ingestion was successful, False otherwise.
    """
    database = get_rag()
    embedding_service = get_embedding_service()

    clean_content = clean_html_text(payload["content"])

    if not clean_content:
        return True

    chunks = chunk_text(clean_content)

    chunk_embeddings = embedding_service.encode(chunks, normalize_embeddings=True)

    points = []
    part_number = 0
    for chunk, embedding in zip(chunks, chunk_embeddings):
        part_number += 1
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"text": chunk, "source_id": str(payload["resourceId"]), "project_id": str(payload["projectId"]), "part_number": part_number}
            )
        )

    if points:
        database.upsert_points(points=points)

    return {"success": True}
