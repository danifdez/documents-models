from utils.job_registry import job_handler
from rag.pipeline import create_ask_pipeline
from rag.types import RAGContext
from config import RAG_DEFAULT_LIMIT, RAG_MAX_TOKENS, RAG_SCORE_THRESHOLD


@job_handler("ask")
def ask_question(payload) -> dict:
    ctx = RAGContext(
        query=payload["question"],
        project_id=str(payload["projectId"]) if payload.get("projectId") else None,
        limit=RAG_DEFAULT_LIMIT,
        max_tokens=RAG_MAX_TOKENS,
        score_threshold=RAG_SCORE_THRESHOLD,
    )

    pipeline = create_ask_pipeline()
    ctx = pipeline.run(ctx)

    if not ctx.response:
        return {"response": "No relevant information was found to answer this question."}

    return {"response": ctx.response}
