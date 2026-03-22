from services.llm_service import get_llm_service
from services.model_config import get_llm_params
from rag.types import RAGContext


class Generator:
    """LLM inference module. Generates a response from the built prompt."""

    def run(self, ctx: RAGContext) -> RAGContext:
        params = get_llm_params("ask")
        llm = get_llm_service(**params)
        ctx.response = llm.generate(ctx.prompt, max_tokens=ctx.max_tokens)
        return ctx
