from rag.types import RAGContext


class Reranker:
    """Post-retrieval module. Filters empty chunks, deduplicates, and sorts by score."""

    def run(self, ctx: RAGContext) -> RAGContext:
        seen = set()
        unique = []

        for chunk in ctx.chunks:
            if not chunk.text.strip():
                continue
            if chunk.text in seen:
                continue
            seen.add(chunk.text)
            unique.append(chunk)

        ctx.ranked_chunks = sorted(unique, key=lambda c: c.score, reverse=True)
        return ctx
