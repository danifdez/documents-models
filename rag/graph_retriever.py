import logging
from rag.types import RAGContext
from database.neo4j_db import get_neo4j
from config import NEO4J_ENABLED

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves entity relationships from Neo4j to enrich RAG context.

    Uses the shared multilingual LLM NER on the query to identify entity names,
    then queries Neo4j for relationships in their neighborhood.
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
        if not NEO4J_ENABLED:
            return ctx

        neo4j = get_neo4j()
        if not neo4j:
            return ctx

        try:
            entity_names = self._extract_entity_names(ctx.query)
            if not entity_names:
                return ctx

            triples = neo4j.query_neighborhood(
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
