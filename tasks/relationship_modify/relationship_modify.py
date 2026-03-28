import logging
from utils.job_registry import job_handler
from database.neo4j_db import get_neo4j

logger = logging.getLogger(__name__)


@job_handler("relationship-modify")
def modify_relationship(payload) -> dict:
    action = payload.get("action")
    subject_id = payload.get("subject_id") or payload.get("subjectId")
    object_id = payload.get("object_id") or payload.get("objectId")
    predicate = payload.get("predicate")
    new_predicate = payload.get("new_predicate") or payload.get("newPredicate")
    resource_id = payload.get("resource_id") or payload.get("resourceId")
    project_id = payload.get("project_id") or payload.get("projectId")

    neo4j = get_neo4j()
    if not neo4j:
        return {"success": False, "action": action, "error": "Neo4j is not enabled"}

    try:
        if action == "create":
            if not all([subject_id, predicate, object_id, resource_id]):
                return {"success": False, "action": action, "error": "Missing required fields"}
            neo4j.create_relationship(
                subject_id=int(subject_id),
                predicate=predicate,
                object_id=int(object_id),
                resource_id=int(resource_id),
                project_id=int(project_id) if project_id else None,
            )

        elif action == "update":
            if not all([subject_id, predicate, object_id, new_predicate, resource_id]):
                return {"success": False, "action": action, "error": "Missing required fields"}
            neo4j.update_relationship(
                subject_id=int(subject_id),
                old_predicate=predicate,
                object_id=int(object_id),
                new_predicate=new_predicate,
                resource_id=int(resource_id),
            )

        elif action == "delete":
            if not all([subject_id, predicate, object_id, resource_id]):
                return {"success": False, "action": action, "error": "Missing required fields"}
            neo4j.delete_relationship(
                subject_id=int(subject_id),
                predicate=predicate,
                object_id=int(object_id),
                resource_id=int(resource_id),
            )

        elif action == "delete-by-resource":
            if not resource_id:
                return {"success": False, "action": action, "error": "Missing resource_id"}
            neo4j.delete_by_resource(int(resource_id))

        else:
            return {"success": False, "action": action, "error": f"Unknown action: {action}"}

        logger.info("Relationship %s completed successfully", action)
        return {"success": True, "action": action}

    except Exception as e:
        logger.error("Error modifying relationship (%s): %s", action, e)
        return {"success": False, "action": action, "error": str(e)}
