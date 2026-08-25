from common.execution_registry import execution_handler
from rag.retriever import Retriever
from rag.reranker import Reranker
from rag.types import RAGContext
from common.vector_contract import load_vector_candidates


@execution_handler("search")
def search_snippets(payload) -> dict:
    ctx = RAGContext(
        query=payload["query"],
        project_id=str(payload["projectId"]) if payload.get("projectId") else None,
        limit=payload["limit"],
        score_threshold=payload.get("score_threshold"),
        candidates=load_vector_candidates(payload),
    )

    ctx = Retriever().run(ctx)
    ctx = Reranker().run(ctx)

    source = ctx.ranked_chunks if ctx.ranked_chunks else ctx.chunks
    return {"results": [
        {"text": c.text, "score": c.score, "metadata": c.metadata}
        for c in source
    ]}
