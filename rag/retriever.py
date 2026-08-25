from common.vector_contract import rank_vector_candidates
from rag.types import RAGContext, RetrievedChunk
from services.embedding_service import get_embedding_service


class Retriever:
    """Ranks the vector candidates frozen into the assignment."""

    def run(self, ctx: RAGContext) -> RAGContext:
        query_embedding = get_embedding_service().encode_query(ctx.query).tolist()
        points = rank_vector_candidates(
            query_embedding,
            ctx.candidates,
            ctx.limit,
            ctx.score_threshold,
        )
        ctx.chunks = [
            RetrievedChunk(
                text=str(point["payload"].get("text") or ""),
                score=point["score"],
                metadata=point["payload"],
            )
            for point in points
        ]
        return ctx
