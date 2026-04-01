import logging
from neo4j import GraphDatabase
from typing import List, Optional, Dict, Any
from config import NEO4J_ENABLED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


class Neo4jDB:

    def __init__(self):
        logger.info("Connecting to Neo4j at %s", NEO4J_URI)
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self._ensure_constraints()

    def _ensure_constraints(self):
        """Create uniqueness constraints and indexes."""
        try:
            with self.driver.session() as session:
                session.run(
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
                )
                session.run(
                    "CREATE INDEX entity_project_id IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.project_id)"
                )
            logger.info("Neo4j constraints and indexes ensured")
        except Exception as e:
            logger.error("Error ensuring Neo4j constraints: %s", e)
            raise

    def close(self):
        self.driver.close()

    def upsert_entity(self, entity_id: int, name: str, entity_type: str,
                       project_id: Optional[int] = None, resource_id: Optional[int] = None):
        """Create or update an entity node."""
        with self.driver.session() as session:
            session.run(
                "MERGE (e:Entity {entity_id: $entity_id}) "
                "SET e.name = $name, e.entity_type = $entity_type, "
                "e.project_id = $project_id, e.resource_id = $resource_id",
                entity_id=entity_id, name=name, entity_type=entity_type,
                project_id=project_id, resource_id=resource_id,
            )

    def upsert_relationship(self, subject_id: int, predicate: str, object_id: int,
                             resource_id: int, project_id: Optional[int] = None,
                             confidence: float = 1.0, context: str = ""):
        """Create or update a relationship between two entity nodes."""
        with self.driver.session() as session:
            session.run(
                "MATCH (s:Entity {entity_id: $subject_id}) "
                "MATCH (o:Entity {entity_id: $object_id}) "
                "MERGE (s)-[r:REL {predicate: $predicate, resource_id: $resource_id}]->(o) "
                "SET r.project_id = $project_id, r.confidence = $confidence, r.context = $context",
                subject_id=subject_id, object_id=object_id,
                predicate=predicate, resource_id=resource_id,
                project_id=project_id, confidence=confidence, context=context,
            )

    def query_by_resource(self, resource_id: int, limit: int = 100) -> Dict[str, Any]:
        """Query all entities and relationships for a given resource."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r:REL {resource_id: $resource_id}]->(o:Entity) "
                "RETURN s.entity_id AS source_id, s.name AS source_name, s.entity_type AS source_type, "
                "r.predicate AS predicate, r.confidence AS confidence, r.resource_id AS resource_id, "
                "o.entity_id AS target_id, o.name AS target_name, o.entity_type AS target_type "
                "LIMIT $limit",
                resource_id=resource_id, limit=limit,
            )
            return self._parse_relationship_results(result)

    def query_by_project(self, project_id: int, resource_ids: Optional[List[int]] = None,
                          limit: int = 200) -> Dict[str, Any]:
        """Query all entities and relationships for a project, optionally filtered by resources."""
        with self.driver.session() as session:
            if resource_ids:
                result = session.run(
                    "MATCH (s:Entity)-[r:REL]->(o:Entity) "
                    "WHERE r.project_id = $project_id AND r.resource_id IN $resource_ids "
                    "RETURN s.entity_id AS source_id, s.name AS source_name, s.entity_type AS source_type, "
                    "r.predicate AS predicate, r.confidence AS confidence, r.resource_id AS resource_id, "
                    "o.entity_id AS target_id, o.name AS target_name, o.entity_type AS target_type "
                    "LIMIT $limit",
                    project_id=project_id, resource_ids=resource_ids, limit=limit,
                )
            else:
                result = session.run(
                    "MATCH (s:Entity)-[r:REL]->(o:Entity) "
                    "WHERE r.project_id = $project_id "
                    "RETURN s.entity_id AS source_id, s.name AS source_name, s.entity_type AS source_type, "
                    "r.predicate AS predicate, r.confidence AS confidence, r.resource_id AS resource_id, "
                    "o.entity_id AS target_id, o.name AS target_name, o.entity_type AS target_type "
                    "LIMIT $limit",
                    project_id=project_id, limit=limit,
                )
            return self._parse_relationship_results(result)

    def query_all(self, limit: int = 500) -> Dict[str, Any]:
        """Query all entities and relationships (global view, no project filter)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r:REL]->(o:Entity) "
                "RETURN s.entity_id AS source_id, s.name AS source_name, s.entity_type AS source_type, "
                "r.predicate AS predicate, r.confidence AS confidence, r.resource_id AS resource_id, "
                "o.entity_id AS target_id, o.name AS target_name, o.entity_type AS target_type "
                "LIMIT $limit",
                limit=limit,
            )
            return self._parse_relationship_results(result)

    def query_neighborhood(self, entity_names: List[str], project_id: Optional[str] = None,
                            depth: int = 2) -> List[Dict[str, Any]]:
        """Query relationships in the neighborhood of given entity names. Used for RAG context."""
        if not entity_names:
            return []

        with self.driver.session() as session:
            if project_id:
                result = session.run(
                    "MATCH (s:Entity)-[r:REL]->(o:Entity) "
                    "WHERE (s.name IN $names OR o.name IN $names) "
                    "AND r.project_id = $project_id "
                    "RETURN s.name AS source, r.predicate AS predicate, o.name AS target, "
                    "r.confidence AS confidence "
                    "LIMIT 20",
                    names=entity_names, project_id=int(project_id),
                )
            else:
                result = session.run(
                    "MATCH (s:Entity)-[r:REL]->(o:Entity) "
                    "WHERE s.name IN $names OR o.name IN $names "
                    "RETURN s.name AS source, r.predicate AS predicate, o.name AS target, "
                    "r.confidence AS confidence "
                    "LIMIT 20",
                    names=entity_names,
                )
            return [dict(record) for record in result]

    def create_relationship(self, subject_id: int, predicate: str, object_id: int,
                             resource_id: int, project_id: Optional[int] = None):
        """Create a new relationship between two existing entities."""
        self.upsert_relationship(subject_id, predicate, object_id, resource_id,
                                  project_id=project_id)

    def update_relationship(self, subject_id: int, old_predicate: str, object_id: int,
                             new_predicate: str, resource_id: int):
        """Update the predicate of an existing relationship."""
        with self.driver.session() as session:
            session.run(
                "MATCH (s:Entity {entity_id: $subject_id})"
                "-[r:REL {predicate: $old_predicate, resource_id: $resource_id}]->"
                "(o:Entity {entity_id: $object_id}) "
                "SET r.predicate = $new_predicate",
                subject_id=subject_id, object_id=object_id,
                old_predicate=old_predicate, new_predicate=new_predicate,
                resource_id=resource_id,
            )

    def delete_relationship(self, subject_id: int, predicate: str, object_id: int,
                             resource_id: int):
        """Delete a specific relationship."""
        with self.driver.session() as session:
            session.run(
                "MATCH (s:Entity {entity_id: $subject_id})"
                "-[r:REL {predicate: $predicate, resource_id: $resource_id}]->"
                "(o:Entity {entity_id: $object_id}) "
                "DELETE r",
                subject_id=subject_id, object_id=object_id,
                predicate=predicate, resource_id=resource_id,
            )

    def delete_by_resource(self, resource_id: int):
        """Delete all relationships associated with a resource."""
        with self.driver.session() as session:
            session.run(
                "MATCH ()-[r:REL {resource_id: $resource_id}]->() DELETE r",
                resource_id=resource_id,
            )
            logger.info("Deleted all relationships for resource %s", resource_id)

    def _parse_relationship_results(self, result) -> Dict[str, Any]:
        """Parse Neo4j query results into entities and relationships format."""
        entities_map = {}
        relationships = []

        for record in result:
            src_id = record["source_id"]
            tgt_id = record["target_id"]

            if src_id not in entities_map:
                entities_map[src_id] = {
                    "id": src_id,
                    "name": record["source_name"],
                    "type": record["source_type"],
                }
            if tgt_id not in entities_map:
                entities_map[tgt_id] = {
                    "id": tgt_id,
                    "name": record["target_name"],
                    "type": record["target_type"],
                }

            relationships.append({
                "source": src_id,
                "target": tgt_id,
                "predicate": record["predicate"],
                "confidence": record["confidence"],
                "resource_id": record["resource_id"],
            })

        return {
            "entities": list(entities_map.values()),
            "relationships": relationships,
        }


# Singleton instance
_neo4j_db = None


def get_neo4j() -> Optional[Neo4jDB]:
    """Get the singleton Neo4j service instance. Returns None if disabled."""
    global _neo4j_db
    if not NEO4J_ENABLED:
        return None
    if _neo4j_db is None:
        _neo4j_db = Neo4jDB()
    return _neo4j_db
