"""Self-contained steps for the durable relationship-extraction workflow."""

import json
import re
from typing import Any, Dict, List, Set

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.grammars import RELATIONSHIPS_GBNF
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs
from services.llm_service import get_llm_service


def _entity_names(entities: Any) -> Set[str]:
    if not isinstance(entities, list) or len(entities) < 2:
        raise ValueError(
            "relationship-extraction-map requires at least two entities"
        )
    names: Set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("relationship-extraction-map entities must be objects")
        name = entity.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("relationship-extraction-map entity names are required")
        names.add(name.strip())
    if len(names) < 2:
        raise ValueError(
            "relationship-extraction-map requires two distinct entity names"
        )
    return names


def _parse_response(response: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(response)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "relationship-extraction-map returned invalid JSON"
        ) from error
    if not isinstance(parsed, list):
        raise ValueError("relationship-extraction-map result must be an array")
    return parsed


def _normalize_predicate(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.casefold())).strip(
        "_"
    )


def _context(content: str, subject: str, obj: str) -> str:
    spans = []
    for name in (subject, obj):
        position = content.casefold().find(name.casefold())
        if position >= 0:
            spans.append((position, position + len(name)))
    if not spans:
        return ""
    first = min(start for start, _ in spans)
    last = max(end for _, end in spans)
    boundaries = ".!?\n"
    left = max(content.rfind(marker, 0, first) for marker in boundaries)
    right_candidates = [
        position
        for marker in boundaries
        if (position := content.find(marker, last)) >= 0
    ]
    start = left + 1 if left >= 0 else max(0, first - 150)
    end = min(right_candidates) + 1 if right_candidates else min(
        len(content), last + 150
    )
    return content[start:end].strip()[:500]


def _validated_relationships(
    entries: List[Dict[str, Any]],
    entity_names: Set[str],
    content: str,
) -> List[Dict[str, Any]]:
    relationships = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        subject = entry.get("subject")
        obj = entry.get("object")
        predicate_value = entry.get("predicate")
        if (
            not isinstance(subject, str)
            or not isinstance(obj, str)
            or not isinstance(predicate_value, str)
            or subject not in entity_names
            or obj not in entity_names
            or subject == obj
        ):
            continue
        predicate = _normalize_predicate(predicate_value)
        if not predicate:
            continue
        relationships.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": 0.5,
                "context": _context(content, subject, obj),
            }
        )
    return relationships


def _validate_partial(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("relationship-extraction-reduce received an invalid entry")
    subject = entry.get("subject")
    predicate = entry.get("predicate")
    obj = entry.get("object")
    confidence = entry.get("confidence")
    context = entry.get("context", "")
    if (
        not isinstance(subject, str)
        or not subject.strip()
        or not isinstance(predicate, str)
        or not predicate.strip()
        or not isinstance(obj, str)
        or not obj.strip()
        or subject == obj
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
        or not isinstance(context, str)
    ):
        raise ValueError("relationship-extraction-reduce received an invalid entry")
    return {
        "subject": subject.strip(),
        "predicate": predicate.strip(),
        "object": obj.strip(),
        "confidence": float(confidence),
        "context": context[:500],
    }


@execution_handler("relationship-extraction-map")
def relationship_extraction_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("relationship-extraction-map content must be a string")
    safe_content = strip_dense_blobs(content).strip()
    if not safe_content:
        raise ValueError("relationship-extraction-map requires non-empty content")
    config = get_task_config("relationship-extraction-map")
    if len(safe_content.split()) > int(config.get("max_input_words", 1500)):
        raise ValueError("relationship-extraction-map content exceeds its word budget")
    entities = payload.get("entities")
    entity_names = _entity_names(entities)
    if len(entities) > int(config.get("max_entities", 200)):
        raise ValueError("relationship-extraction-map has too many entities")
    if sum(len(name.split()) for name in entity_names) > int(
        config.get("max_entity_words", 500)
    ):
        raise ValueError("relationship-extraction-map entities exceed their budget")
    prompt_template = get_prompt("relationship-extraction-map")
    if not prompt_template:
        raise RuntimeError("relationship-extraction-map prompt is unavailable")
    entity_block = "\n".join(
        f"- {entity['name']} ({entity.get('type', 'UNKNOWN')})"
        for entity in entities
    )
    response = get_llm_service(
        **get_llm_params("relationship-extraction-map")
    ).chat(
        [
            {
                "role": "user",
                "content": prompt_template.format(
                    entities=entity_block,
                    text=safe_content,
                ),
            }
        ],
        max_tokens=int(config.get("max_tokens", 2000)),
        grammar=RELATIONSHIPS_GBNF,
        temperature=0.0,
    )
    return {
        "relationships": _validated_relationships(
            _parse_response(response),
            entity_names,
            safe_content,
        )
    }


@execution_handler("relationship-extraction-reduce")
def relationship_extraction_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("relationship-extraction-reduce requires partials")
    best: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for partial in partials:
        if not isinstance(partial, list):
            raise ValueError("relationship-extraction-reduce partials must be arrays")
        for raw_entry in partial:
            entry = _validate_partial(raw_entry)
            key = (entry["subject"], entry["predicate"], entry["object"])
            if key not in best:
                order.append(key)
                best[key] = entry
            elif entry["confidence"] > best[key]["confidence"]:
                best[key] = entry
    return {"relationships": [best[key] for key in order]}
