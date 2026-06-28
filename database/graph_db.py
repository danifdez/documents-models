import json
import logging
import psycopg
from psycopg.rows import tuple_row
from typing import List, Optional, Dict, Any

from config import (
    GRAPH_ENABLED,
    GRAPH_NAME,
    GRAPH_NEIGHBORHOOD_DEPTH,
    GRAPH_NEIGHBORHOOD_LIMIT,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)

logger = logging.getLogger(__name__)


def _agval(v: Any) -> Any:
    """Decode a scalar agtype cell into a native Python value.

    AGE returns scalars as JSON-compatible text (e.g. '123', '"name"', 'null'),
    which psycopg hands back as a plain string for the unknown agtype OID.
    """
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v


class GraphDB:
    """Entity/relationship graph backed by Apache AGE (openCypher over PostgreSQL).

    Drop-in replacement for the former Neo4j driver: same public method surface,
    so the relationship tasks and the RAG GraphRetriever need no logic changes.
    The graph shares the application's PostgreSQL instance — no separate service.
    """

    def __init__(self):
        logger.info("Connecting to graph (Apache AGE) graph=%s db=%s", GRAPH_NAME, POSTGRES_DB)
        self.conn = self._connect()
        self._ensure_graph()
        self._ensure_indexes()

    def _connect(self):
        conn = psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            autocommit=True,
            row_factory=tuple_row,
        )
        # AGE must be loaded and its catalog put on the search_path per session
        # before any cypher() call.
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            cur.execute("LOAD 'age'")
            cur.execute('SET search_path = ag_catalog, "$user", public')
        return conn

    def _ensure_graph(self):
        """Create the graph if it doesn't exist yet (idempotent)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s",
                    (GRAPH_NAME,),
                )
                exists = cur.fetchone()[0]
                if not exists:
                    cur.execute("SELECT create_graph(%s)", (GRAPH_NAME,))
                    logger.info("Created AGE graph '%s'", GRAPH_NAME)
        except Exception as e:
            logger.error("Error ensuring AGE graph: %s", e)
            raise

    def _ensure_indexes(self):
        """Index the entity_id property on the vertex table for MERGE lookups.

        AGE has no Neo4j-style CREATE CONSTRAINT; uniqueness is enforced via MERGE
        (as before). A btree on the property column keeps lookups fast.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f'CREATE INDEX IF NOT EXISTS entity_entity_id_idx '
                    f'ON {GRAPH_NAME}."Entity" '
                    f"USING btree (ag_catalog.agtype_access_operator(properties, '\"entity_id\"'::agtype))"
                )
            logger.info("AGE indexes ensured")
        except Exception as e:
            # Non-fatal: the graph works without the index, just slower.
            logger.warning("Could not ensure AGE indexes: %s", e)

    def close(self):
        self.conn.close()

    # ── internal cypher execution ────────────────────────────────────────────

    def _cypher(self, body: str, params: Optional[Dict[str, Any]], columns: str) -> List[tuple]:
        """Run a Cypher statement and return raw rows.

        `body` is a static, developer-authored string with Cypher `$name`
        placeholders. Values are passed via AGE's single-agtype-parameter
        convention (the third argument of cypher()), NOT interpolated — this is
        what keeps the call injection-safe. `columns` declares the RETURN shape,
        e.g. "source agtype, predicate agtype".
        """
        sql = f"SELECT * FROM cypher('{GRAPH_NAME}', $$ {body} $$, %s) AS ({columns})"
        with self.conn.cursor() as cur:
            cur.execute(sql, (json.dumps(params or {}),))
            return cur.fetchall()

    def _exec(self, body: str, params: Optional[Dict[str, Any]]) -> None:
        """Run a write Cypher statement that returns nothing.

        AGE still requires a column definition list, so we RETURN a dummy value.
        """
        sql = f"SELECT * FROM cypher('{GRAPH_NAME}', $$ {body} RETURN 1 $$, %s) AS (ok agtype)"
        with self.conn.cursor() as cur:
            cur.execute(sql, (json.dumps(params or {}),))

    # ── writes ───────────────────────────────────────────────────────────────

    def upsert_entity(self, entity_id: int, name: str, entity_type: str,
                      project_id: Optional[int] = None, resource_id: Optional[int] = None):
        """Create or update an entity node."""
        self._exec(
            "MERGE (e:Entity {entity_id: $entity_id}) "
            "SET e.name = $name, e.entity_type = $entity_type, "
            "e.project_id = $project_id, e.resource_id = $resource_id",
            {
                "entity_id": entity_id, "name": name, "entity_type": entity_type,
                "project_id": project_id, "resource_id": resource_id,
            },
        )

    def upsert_relationship(self, subject_id: int, predicate: str, object_id: int,
                            resource_id: int, project_id: Optional[int] = None,
                            confidence: float = 1.0, context: str = ""):
        """Create or update a relationship between two entity nodes."""
        self._exec(
            "MATCH (s:Entity {entity_id: $subject_id}) "
            "MATCH (o:Entity {entity_id: $object_id}) "
            "MERGE (s)-[r:REL {predicate: $predicate, resource_id: $resource_id}]->(o) "
            "SET r.project_id = $project_id, r.confidence = $confidence, r.context = $context",
            {
                "subject_id": subject_id, "object_id": object_id,
                "predicate": predicate, "resource_id": resource_id,
                "project_id": project_id, "confidence": confidence, "context": context,
            },
        )

    # ── reads (1-hop inventory views) ─────────────────────────────────────────

    _TRIPLE_RETURN = (
        "RETURN s.entity_id AS source_id, s.name AS source_name, s.entity_type AS source_type, "
        "r.predicate AS predicate, r.confidence AS confidence, r.resource_id AS resource_id, "
        "o.entity_id AS target_id, o.name AS target_name, o.entity_type AS target_type"
    )
    _TRIPLE_COLS = (
        "source_id agtype, source_name agtype, source_type agtype, "
        "predicate agtype, confidence agtype, resource_id agtype, "
        "target_id agtype, target_name agtype, target_type agtype"
    )

    def query_by_resource(self, resource_id: int, limit: int = 100) -> Dict[str, Any]:
        """Query all entities and relationships for a given resource."""
        rows = self._cypher(
            "MATCH (s:Entity)-[r:REL {resource_id: $resource_id}]->(o:Entity) "
            f"{self._TRIPLE_RETURN} LIMIT {int(limit)}",
            {"resource_id": resource_id},
            self._TRIPLE_COLS,
        )
        return self._parse_triple_rows(rows)

    def query_by_project(self, project_id: int, resource_ids: Optional[List[int]] = None,
                         limit: int = 200) -> Dict[str, Any]:
        """Query all entities and relationships for a project, optionally filtered by resources."""
        if resource_ids:
            body = (
                "MATCH (s:Entity)-[r:REL]->(o:Entity) "
                "WHERE r.project_id = $project_id AND r.resource_id IN $resource_ids "
                f"{self._TRIPLE_RETURN} LIMIT {int(limit)}"
            )
            params = {"project_id": project_id, "resource_ids": resource_ids}
        else:
            body = (
                "MATCH (s:Entity)-[r:REL]->(o:Entity) "
                "WHERE r.project_id = $project_id "
                f"{self._TRIPLE_RETURN} LIMIT {int(limit)}"
            )
            params = {"project_id": project_id}
        return self._parse_triple_rows(self._cypher(body, params, self._TRIPLE_COLS))

    def query_all(self, limit: int = 500) -> Dict[str, Any]:
        """Query all entities and relationships (global view, no project filter)."""
        rows = self._cypher(
            "MATCH (s:Entity)-[r:REL]->(o:Entity) "
            f"{self._TRIPLE_RETURN} LIMIT {int(limit)}",
            {},
            self._TRIPLE_COLS,
        )
        return self._parse_triple_rows(rows)

    # ── multi-hop neighborhood (GraphRAG core) ────────────────────────────────

    def query_neighborhood(self, entity_names: List[str], project_id: Optional[str] = None,
                           depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """Traverse the relationship subgraph up to `depth` hops out from the
        seed entities. This is what makes the RAG graph-aware: starting from the
        entities named in the question, it walks B→C→D chains, not just direct
        edges. Traversal is undirected to maximise recovered context; each
        returned edge keeps its real orientation via startNode/endNode.
        """
        if not entity_names:
            return []

        hops = int(depth if depth is not None else GRAPH_NEIGHBORHOOD_DEPTH)
        hops = max(1, min(hops, 5))  # clamp: deep traversals can blow up the subgraph
        limit = int(GRAPH_NEIGHBORHOOD_LIMIT)

        project_filter = (
            "AND seed.project_id = $project_id " if project_id is not None else ""
        )
        body = (
            f"MATCH p = (seed:Entity)-[:REL*1..{hops}]-(other:Entity) "
            f"WHERE seed.name IN $names {project_filter}"
            "UNWIND relationships(p) AS r "
            "WITH startNode(r) AS s, r, endNode(r) AS o "
            "RETURN DISTINCT s.name AS source, r.predicate AS predicate, "
            "o.name AS target, r.confidence AS confidence "
            f"LIMIT {limit}"
        )
        params: Dict[str, Any] = {"names": entity_names}
        if project_id is not None:
            params["project_id"] = int(project_id)

        rows = self._cypher(
            body, params,
            "source agtype, predicate agtype, target agtype, confidence agtype",
        )
        results = [
            {
                "source": _agval(src),
                "predicate": _agval(pred),
                "target": _agval(tgt),
                "confidence": _agval(conf),
            }
            for src, pred, tgt, conf in rows
        ]
        if len(results) >= limit:
            logger.info(
                "query_neighborhood hit the %d-edge limit (depth=%d); subgraph truncated",
                limit, hops,
            )
        return results

    # ── targeted mutations ────────────────────────────────────────────────────

    def create_relationship(self, subject_id: int, predicate: str, object_id: int,
                            resource_id: int, project_id: Optional[int] = None):
        """Create a new relationship between two existing entities."""
        self.upsert_relationship(subject_id, predicate, object_id, resource_id,
                                 project_id=project_id)

    def update_relationship(self, subject_id: int, old_predicate: str, object_id: int,
                            new_predicate: str, resource_id: int):
        """Update the predicate of an existing relationship."""
        self._exec(
            "MATCH (s:Entity {entity_id: $subject_id})"
            "-[r:REL {predicate: $old_predicate, resource_id: $resource_id}]->"
            "(o:Entity {entity_id: $object_id}) "
            "SET r.predicate = $new_predicate",
            {
                "subject_id": subject_id, "object_id": object_id,
                "old_predicate": old_predicate, "new_predicate": new_predicate,
                "resource_id": resource_id,
            },
        )

    def delete_relationship(self, subject_id: int, predicate: str, object_id: int,
                            resource_id: int):
        """Delete a specific relationship."""
        self._exec(
            "MATCH (s:Entity {entity_id: $subject_id})"
            "-[r:REL {predicate: $predicate, resource_id: $resource_id}]->"
            "(o:Entity {entity_id: $object_id}) "
            "DELETE r",
            {
                "subject_id": subject_id, "object_id": object_id,
                "predicate": predicate, "resource_id": resource_id,
            },
        )

    def delete_by_resource(self, resource_id: int):
        """Delete all relationships associated with a resource."""
        self._exec(
            "MATCH ()-[r:REL {resource_id: $resource_id}]->() DELETE r",
            {"resource_id": resource_id},
        )
        logger.info("Deleted all relationships for resource %s", resource_id)

    # ── result shaping ─────────────────────────────────────────────────────────

    def _parse_triple_rows(self, rows: List[tuple]) -> Dict[str, Any]:
        """Shape triple rows into the {entities, relationships} dict the backend
        reads verbatim. Column order matches _TRIPLE_RETURN."""
        entities_map: Dict[Any, Dict[str, Any]] = {}
        relationships = []

        for row in rows:
            (source_id, source_name, source_type, predicate, confidence,
             resource_id, target_id, target_name, target_type) = (_agval(c) for c in row)

            if source_id not in entities_map:
                entities_map[source_id] = {
                    "id": source_id, "name": source_name, "type": source_type,
                }
            if target_id not in entities_map:
                entities_map[target_id] = {
                    "id": target_id, "name": target_name, "type": target_type,
                }

            relationships.append({
                "source": source_id,
                "target": target_id,
                "predicate": predicate,
                "confidence": confidence,
                "resource_id": resource_id,
            })

        return {
            "entities": list(entities_map.values()),
            "relationships": relationships,
        }


# Singleton instance
_graph_db = None


def get_graph() -> Optional[GraphDB]:
    """Get the singleton graph service instance. Returns None if disabled."""
    global _graph_db
    if not GRAPH_ENABLED:
        return None
    if _graph_db is None:
        _graph_db = GraphDB()
    return _graph_db
