from services.embedding_service import get_embedding_service
from database.rag import get_rag
from rag.types import RAGContext, RetrievedChunk


class Retriever:
    """Vector search module. Encodes the query and retrieves similar chunks from Qdrant."""

    def run(self, ctx: RAGContext) -> RAGContext:
        embedding_service = get_embedding_service()
        db = get_rag()

        query_embedding = embedding_service.encode_query(ctx.query)

        points = db.query_points(
            query_embedding,
            limit=ctx.limit,
            with_payload=True,
            project_id=ctx.project_id,
            score_threshold=ctx.score_threshold,
        )

        ctx.chunks = [
            RetrievedChunk(
                text=p.payload.get("text", "") if hasattr(p, "payload") else "",
                score=getattr(p, "score", 0.0),
                metadata=p.payload if hasattr(p, "payload") else {},
            )
            for p in points
        ]
        return ctx
