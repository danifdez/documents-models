import json
import re
import logging
from typing import List, Dict, Set
from utils.job_registry import job_handler
from services.llm_service import get_llm_service
from services.prompts import get_prompt
from services.model_config import get_llm_params, get_task_config
from services.text import normalize_text
from database.neo4j_db import get_neo4j

logger = logging.getLogger(__name__)


def _parse_json_array(text: str) -> list:
    """Extract a JSON array from LLM output, handling markdown fences."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _validate_relationships(relationships: list, entity_names: Set[str]) -> list:
    """Filter relationships to only those referencing known entities."""
    valid = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        subject = rel.get("subject", "")
        obj = rel.get("object", "")
        predicate = rel.get("predicate", "")
        if not subject or not obj or not predicate:
            continue
        if subject not in entity_names or obj not in entity_names:
            continue
        if subject == obj:
            continue
        valid.append({
            "subject": subject,
            "predicate": str(predicate).lower().replace(" ", "_"),
            "object": obj,
            "confidence": float(rel.get("confidence", 0.5)),
            "context": str(rel.get("context", ""))[:500],
        })
    return valid


def _deduplicate(relationships: list) -> list:
    """Remove duplicate relationships keeping the highest confidence."""
    best: Dict[tuple, dict] = {}
    for rel in relationships:
        key = (rel["subject"], rel["predicate"], rel["object"])
        if key not in best or rel["confidence"] > best[key]["confidence"]:
            best[key] = rel
    return list(best.values())


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _extract_from_chunk(chunk: str, entities_str: str, prompt_template: str,
                         llm_service, max_tokens: int) -> str:
    """Run LLM on a single chunk and return raw response."""
    prompt = prompt_template.format(entities=entities_str, text=chunk)
    try:
        return llm_service.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
    except Exception:
        return llm_service.generate(prompt, max_tokens=max_tokens)


@job_handler("relationship-extraction")
def extract_relationships(payload) -> dict:
    text = payload.get("text", "")
    entities = payload.get("entities", [])
    resource_id = payload.get("resource_id") or payload.get("resourceId")
    project_id = payload.get("project_id") or payload.get("projectId")

    if not text or not entities:
        return {"relationships": []}

    task_config = get_task_config("relationship-extraction")
    text = normalize_text(str(text))

    # Build entity lookup
    entity_names = {e["name"] for e in entities}
    entity_map = {e["name"]: e for e in entities}
    entities_str = "\n".join(
        f"- {e['name']} ({e.get('type', 'UNKNOWN')})" for e in entities
    )

    # Build prompt template
    prompt_template = get_prompt("relationship-extraction")
    if not prompt_template:
        return {"relationships": [], "error": "Prompt template not found"}

    # Load LLM
    try:
        params = get_llm_params("relationship-extraction")
        llm_service = get_llm_service(**params)
    except Exception as e:
        logger.error("LLM error during relationship extraction: %s", e)
        return {"relationships": [], "error": str(e)}

    max_tokens = task_config.get("max_tokens", 2000)
    chunk_size = task_config.get("chunk_words", 600)
    chunk_overlap = task_config.get("chunk_overlap", 100)

    # Split text into chunks
    chunks = _chunk_text(text, chunk_size, chunk_overlap)
    logger.info("Processing %d chunk(s) for relationship extraction (%d entities, %d chars)",
                len(chunks), len(entities), len(text))

    # Extract from each chunk
    all_relationships = []
    for i, chunk in enumerate(chunks):
        try:
            generated = _extract_from_chunk(chunk, entities_str, prompt_template,
                                             llm_service, max_tokens)
            logger.info("Chunk %d/%d LLM response (%d chars): %s",
                        i + 1, len(chunks), len(generated), generated[:300])

            raw = _parse_json_array(generated)
            valid = _validate_relationships(raw, entity_names)
            logger.info("Chunk %d/%d: %d raw -> %d valid relationships", i + 1, len(chunks), len(raw), len(valid))
            all_relationships.extend(valid)
        except Exception as e:
            logger.error("Error processing chunk %d: %s", i + 1, e)

    # Deduplicate across chunks
    relationships = _deduplicate(all_relationships)
    logger.info("Total: %d relationships after deduplication", len(relationships))

    # Store in Neo4j
    neo4j = get_neo4j()
    if neo4j and relationships:
        try:
            seen_entities = set()
            for rel in relationships:
                for name in (rel["subject"], rel["object"]):
                    if name not in seen_entities:
                        seen_entities.add(name)
                        e = entity_map[name]
                        neo4j.upsert_entity(
                            entity_id=e["id"],
                            name=e["name"],
                            entity_type=e.get("type", "UNKNOWN"),
                            project_id=int(project_id) if project_id else None,
                            resource_id=int(resource_id) if resource_id else None,
                        )

            for rel in relationships:
                subject = entity_map[rel["subject"]]
                obj = entity_map[rel["object"]]
                neo4j.upsert_relationship(
                    subject_id=subject["id"],
                    predicate=rel["predicate"],
                    object_id=obj["id"],
                    resource_id=int(resource_id) if resource_id else 0,
                    project_id=int(project_id) if project_id else None,
                    confidence=rel["confidence"],
                    context=rel.get("context", ""),
                )

            logger.info("Stored %d relationships for resource %s in Neo4j",
                        len(relationships), resource_id)
        except Exception as e:
            logger.error("Failed to store relationships in Neo4j: %s", e)
            return {"relationships": relationships, "error": f"Neo4j storage failed: {e}"}

    return {
        "relationships": [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"],
            }
            for r in relationships
        ],
        "resourceId": resource_id,
    }
