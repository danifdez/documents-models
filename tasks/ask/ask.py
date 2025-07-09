from utils.job_registry import job_handler
from services.embedding_service import get_embedding_service
from database.rag import get_rag
from llama_cpp import Llama

@job_handler("ask")
def ask_question(payload) -> dict:
    LLM_PATH = "/app/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    llm = Llama(model_path=LLM_PATH, n_ctx=32768, n_threads=4, n_batch=64)

    embedding_service = get_embedding_service()
    db = get_rag()
    query_embedding = embedding_service.encode_single(payload["question"])
    points = db.query_points(query_embedding, limit=5, with_payload=True)
    context = ""
    for point in points:
        text = point.payload.get("text", "") if hasattr(point, 'payload') else ""
        context += "\n" + text

    prompt = f"Answer the following question using only the information provided. If necessary, translate the text to respond in the language the question is asked. Use a maximum of 1000 tokens.\n\nContext:\n{context}\n\nQuestion: {payload['question']}\n\nAnswer:"

    response = llm(prompt, max_tokens=1000, echo=False)

    return { "response": response["choices"][0]["text"].strip() }