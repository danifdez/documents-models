from services.llm_service import get_llm_service
from lib.llm.config import get_llm_params
from rag.types import RAGContext


class Generator:
    """LLM inference module. Generates a response from the built prompt."""

    def run(self, ctx: RAGContext) -> RAGContext:
        params = get_llm_params("ask")
        llm = get_llm_service(**params)
        # `ask` y no `generate`: el prompt es una instrucción con el contexto
        # recuperado dentro, y una completion cruda invita al modelo a seguir
        # escribiendo el contexto en vez de responder la pregunta.
        ctx.response = llm.ask(ctx.prompt, max_tokens=ctx.max_tokens)
        return ctx
