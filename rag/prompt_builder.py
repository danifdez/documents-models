from rag.types import RAGContext
from services.prompts import get_prompt


class PromptBuilder:
    """Builds the final LLM prompt from context and query."""

    def run(self, ctx: RAGContext) -> RAGContext:
        template = get_prompt("ask")
        ctx.prompt = template.format(
            max_tokens=ctx.max_tokens,
            context=ctx.context_text,
            question=ctx.query,
        )
        return ctx
