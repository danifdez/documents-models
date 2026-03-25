from utils.job_registry import job_handler
from rag.pipeline import create_ask_pipeline
from rag.types import RAGContext
from services.model_config import get_rag_config, get_task_config


@job_handler("ask")
def ask_question(payload) -> dict:
    rag = get_rag_config()
    task = get_task_config("ask")

    ctx = RAGContext(
        query=payload["question"],
        project_id=str(payload["projectId"]) if payload.get("projectId") else None,
        limit=task.get("rag_default_limit", rag.get("default_limit", 5)),
        max_tokens=task.get("rag_max_tokens", rag.get("max_tokens", 1000)),
        score_threshold=task.get("rag_score_threshold", rag.get("score_threshold", 0.35)),
    )

    pipeline = create_ask_pipeline()
    ctx = pipeline.run(ctx)

    if not ctx.response:
        return {"response": "No relevant information was found to answer this question."}

    return {"response": ctx.response}
