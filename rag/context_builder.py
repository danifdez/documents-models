from rag.types import RAGContext


class ContextBuilder:
    """Assembles ranked chunks and graph context into a single context string for the LLM."""

    def __init__(self, separator="\n\n---\n\n"):
        self.separator = separator

    def run(self, ctx: RAGContext) -> RAGContext:
        source = ctx.ranked_chunks if ctx.ranked_chunks else ctx.chunks
        parts = [c.text for c in source if c.text]

        # Append graph context if available
        if ctx.graph_context:
            lines = []
            for triple in ctx.graph_context:
                src = triple.get("source", "")
                pred = triple.get("predicate", "")
                tgt = triple.get("target", "")
                if src and pred and tgt:
                    lines.append(f"- {src} --[{pred}]--> {tgt}")
            if lines:
                graph_section = "Entity Relationships:\n" + "\n".join(lines)
                parts.append(graph_section)

        ctx.context_text = self.separator.join(parts)
        return ctx
