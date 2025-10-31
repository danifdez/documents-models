from utils.job_registry import job_handler
from services.embedding_service import get_embedding_service
from database.rag import get_rag
from llama_cpp import Llama
from config import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_N_BATCH,
    RAG_DEFAULT_LIMIT,
    RAG_MAX_TOKENS
)


@job_handler("ask")
def ask_question(payload) -> dict:
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_batch=LLM_N_BATCH
    )

    embedding_service = get_embedding_service()
    db = get_rag()
    query_embedding = embedding_service.encode_single(payload["question"])
    points = db.query_points(
        query_embedding,
        limit=RAG_DEFAULT_LIMIT,
        with_payload=True
    )
    context = ""
    for point in points:
        text = point.payload.get("text", "") if hasattr(
            point, 'payload') else ""
        context += "\n" + text

    prompt = f"Answer the following question using only the information provided. If necessary, translate the text to respond in the language the question is asked. Use a maximum of {RAG_MAX_TOKENS} tokens.\n\nContext:\n{context}\n\nQuestion: {payload['question']}\n\nAnswer:"

    response = llm(prompt, max_tokens=RAG_MAX_TOKENS, echo=False)

    return {"response": response["choices"][0]["text"].strip()}
