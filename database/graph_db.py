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
    """Read-only Apache AGE client used by GraphRAG retrieval."""

    def __init__(self):
        logger.info("Connecting to graph (Apache AGE) graph=%s db=%s", GRAPH_NAME, POSTGRES_DB)
        self.conn = self._connect()

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
            cur.execute("LOAD 'age'")
            cur.execute('SET search_path = ag_catalog, "$user", public')
        return conn

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

        relationship_pattern = (
            f":REL*1..{hops} {{project_id: $project_id}}"
            if project_id is not None else f":REL*1..{hops}"
        )
        body = (
            f"MATCH p = (seed:Entity)-[{relationship_pattern}]-(other:Entity) "
            "WHERE seed.name IN $names "
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
