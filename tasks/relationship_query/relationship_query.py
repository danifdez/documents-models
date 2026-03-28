import logging
from utils.job_registry import job_handler
from database.neo4j_db import get_neo4j

logger = logging.getLogger(__name__)


@job_handler("relationship-query")
def query_relationships(payload) -> dict:
    query_type = payload.get("query_type", "by-resource")
    resource_id = payload.get("resource_id") or payload.get("resourceId")
    project_id = payload.get("project_id") or payload.get("projectId")
    resource_ids = payload.get("resource_ids") or payload.get("resourceIds")
    entity_names = payload.get("entity_names") or payload.get("entityNames")
    limit = payload.get("limit", 200)

    neo4j = get_neo4j()
    if not neo4j:
        return {"entities": [], "relationships": [], "error": "Neo4j is not enabled"}

    try:
        if query_type == "by-resource" and resource_id:
            result = neo4j.query_by_resource(int(resource_id), limit=limit)
        elif query_type == "by-project" and project_id:
            rid_list = [int(r) for r in resource_ids] if resource_ids else None
            result = neo4j.query_by_project(int(project_id), resource_ids=rid_list, limit=limit)
        elif query_type == "neighborhood" and entity_names:
            triples = neo4j.query_neighborhood(
                entity_names,
                project_id=str(project_id) if project_id else None,
            )
            # Convert neighborhood format to standard format
            entities_map = {}
            relationships = []
            for t in triples:
                for name in (t["source"], t["target"]):
                    if name not in entities_map:
                        entities_map[name] = {"id": name, "name": name, "type": ""}
                relationships.append({
                    "source": t["source"],
                    "target": t["target"],
                    "predicate": t["predicate"],
                    "confidence": t.get("confidence", 1.0),
                })
            result = {"entities": list(entities_map.values()), "relationships": relationships}
        else:
            return {"entities": [], "relationships": [], "error": f"Invalid query_type or missing params: {query_type}"}

        return result

    except Exception as e:
        logger.error("Error querying relationships: %s", e)
        return {"entities": [], "relationships": [], "error": str(e)}
