from typing import List, Any
from rag.types import RAGContext
from rag.retriever import Retriever
from rag.reranker import Reranker
from rag.context_builder import ContextBuilder
from rag.prompt_builder import PromptBuilder
from rag.generator import Generator


class RAGPipeline:
    """Chains a list of stage modules, each with a .run(ctx) method."""

    def __init__(self, stages: List[Any]):
        self.stages = stages

    def run(self, ctx: RAGContext) -> RAGContext:
        for stage in self.stages:
            ctx = stage.run(ctx)
        return ctx


def create_ask_pipeline() -> RAGPipeline:
    """Full RAG pipeline: retrieve -> rerank -> build context -> build prompt -> generate."""
    return RAGPipeline([
        Retriever(),
        Reranker(),
        ContextBuilder(),
        PromptBuilder(),
        Generator(),
    ])


def create_search_pipeline() -> RAGPipeline:
    """Search-only pipeline: retrieve -> rerank."""
    return RAGPipeline([
        Retriever(),
        Reranker(),
    ])
