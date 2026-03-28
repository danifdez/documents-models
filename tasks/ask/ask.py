import logging
from utils.job_registry import job_handler
from rag.pipeline import create_ask_pipeline
from rag.types import RAGContext
from services.model_config import get_rag_config, get_task_config

logger = logging.getLogger(__name__)


@job_handler("ask")
def ask_question(payload) -> dict:
    rag = get_rag_config()
    task = get_task_config("ask")

    logger.info("ASK payload: question='%s', projectId=%s", payload.get("question"), payload.get("projectId"))

    ctx = RAGContext(
        query=payload["question"],
        project_id=str(payload["projectId"]) if payload.get("projectId") else None,
        limit=task.get("rag_default_limit", rag.get("default_limit", 5)),
        max_tokens=task.get("rag_max_tokens", rag.get("max_tokens", 1000)),
        score_threshold=task.get("rag_score_threshold", rag.get("score_threshold", 0.35)),
    )

    pipeline = create_ask_pipeline()
    ctx = pipeline.run(ctx)

    logger.info("ASK retriever: %d chunks, scores=%s", len(ctx.chunks), [round(c.score, 3) for c in ctx.chunks[:5]])
    logger.info("ASK graph_context: %d triples", len(ctx.graph_context))
    if ctx.graph_context:
        for t in ctx.graph_context[:10]:
            logger.info("  GRAPH: %s -[%s]-> %s", t.get("source"), t.get("predicate"), t.get("target"))
    logger.info("ASK context_text length: %d chars", len(ctx.context_text))
    logger.info("ASK prompt length: %d chars", len(ctx.prompt))
    logger.info("ASK response (%d chars): %s", len(ctx.response), ctx.response[:300] if ctx.response else "(empty)")

    if not ctx.response:
        return {"response": "No relevant information was found to answer this question."}

    return {"response": ctx.response}
