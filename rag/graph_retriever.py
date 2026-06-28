import logging
from rag.types import RAGContext
from database.graph_db import get_graph
from config import GRAPH_ENABLED

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Enriches RAG context with a multi-hop walk of the entity graph (GraphRAG).

    Uses the shared multilingual LLM NER on the query to identify entity names,
    then traverses the relationship subgraph around them (depth from config) so
    the LLM sees chains of related facts, not just the directly-mentioned ones.
    """

    def _extract_entity_names(self, text: str) -> list:
        """Extract entity names from the query via the shared LLM extractor."""
        try:
            from tasks.entities.entities import extract_entity_names
            return extract_entity_names(text)
        except Exception as e:
            logger.warning("Could not extract entities for graph retrieval: %s", e)
            return []

    def run(self, ctx: RAGContext) -> RAGContext:
        if not GRAPH_ENABLED:
            return ctx

        graph = get_graph()
        if not graph:
            return ctx

        try:
            entity_names = self._extract_entity_names(ctx.query)
            if not entity_names:
                return ctx

            triples = graph.query_neighborhood(
                entity_names,
                project_id=ctx.project_id,
            )
            ctx.graph_context = triples
            if triples:
                logger.info(
                    "GraphRetriever found %d relationships for entities: %s",
                    len(triples), entity_names,
                )
        except Exception as e:
            logger.error("GraphRetriever error: %s", e)

        return ctx
