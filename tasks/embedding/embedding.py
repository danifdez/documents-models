from utils.job_registry import job_handler
from services.embedding_service import get_embedding_service

@job_handler("embedding")
def create_embedding(payload) -> dict:
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode_single(payload["text"])

    return { "results": query_embedding.tolist() }
