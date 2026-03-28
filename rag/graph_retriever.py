import logging
from rag.types import RAGContext
from database.neo4j_db import get_neo4j
from config import NEO4J_ENABLED

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves entity relationships from Neo4j to enrich RAG context.

    Uses spaCy NER on the query to identify entity names, then queries
    Neo4j for relationships in their neighborhood.
    """

    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning("Could not load spaCy model for graph retrieval: %s", e)
        return self._nlp

    def _extract_entity_names(self, text: str) -> list:
        """Extract entity names from text using spaCy NER."""
        nlp = self._get_nlp()
        if not nlp:
            return []
        doc = nlp(text)
        return list({ent.text for ent in doc.ents if len(ent.text) >= 2})

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
