from utils.job_registry import job_handler
from rag.retriever import Retriever
from rag.reranker import Reranker
from rag.types import RAGContext


@job_handler("search")
def search_snippets(payload) -> dict:
    ctx = RAGContext(
        query=payload["query"],
        project_id=str(payload["projectId"]) if payload.get("projectId") else None,
        limit=payload["limit"],
        score_threshold=payload.get("score_threshold"),
    )

    ctx = Retriever().run(ctx)
    ctx = Reranker().run(ctx)

    source = ctx.ranked_chunks if ctx.ranked_chunks else ctx.chunks
    return {"results": [
        {"text": c.text, "score": c.score, "metadata": c.metadata}
        for c in source
    ]}
