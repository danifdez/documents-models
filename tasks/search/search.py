from utils.job_registry import job_handler
from services.embedding_service import get_embedding_service
from database.rag import get_rag

@job_handler("search")
def search_snippets(payload) -> dict:
    embedding_service = get_embedding_service()
    db = get_rag()
    query_embedding = embedding_service.encode_single(payload["query"])
    points = db.query_points(query_embedding, limit=payload["limit"], with_payload=True)
    results = []
    for point in points:
        text = point.payload.get("text", "") if hasattr(point, 'payload') else ""
        score = getattr(point, 'score', 0.0)
        metadata = point.payload if hasattr(point, 'payload') else {}
        results.append({ "text": text, "score": score, "metadata": metadata })
    return { "results": results }
