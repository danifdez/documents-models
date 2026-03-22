from rag.types import RAGContext


class ContextBuilder:
    """Assembles ranked chunks into a single context string for the LLM."""

    def __init__(self, separator="\n\n---\n\n"):
        self.separator = separator

    def run(self, ctx: RAGContext) -> RAGContext:
        source = ctx.ranked_chunks if ctx.ranked_chunks else ctx.chunks
        ctx.context_text = self.separator.join(c.text for c in source if c.text)
        return ctx
