"""Self-contained steps for the durable relationship-extraction workflow."""

import re
from itertools import combinations, islice
from math import exp, log
from threading import Lock
from typing import Any, Dict, List, Set

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from lib.llm.text import strip_dense_blobs

_PERSON = ["PERSON"]
_ORGANIZATION = ["ORG", "NORP"]
_PLACE = ["GPE", "LOC"]
_CREATION = ["PRODUCT", "WORK_OF_ART", "LAW", "EVENT"]
_DEFAULT_RELATIONS = [
    {"predicate": "works_for", "label": "works for", "source_types": _PERSON, "target_types": _ORGANIZATION, "priority": 2},
    {"predicate": "leads", "label": "leads", "source_types": _PERSON, "target_types": _ORGANIZATION, "priority": 3},
    {"predicate": "member_of", "label": "is a member of", "source_types": _PERSON + _ORGANIZATION, "target_types": _ORGANIZATION, "priority": 2},
    {"predicate": "founded", "label": "founded", "source_types": _PERSON + _ORGANIZATION, "target_types": _ORGANIZATION, "priority": 3},
    {"predicate": "owns", "label": "owns", "source_types": _PERSON + _ORGANIZATION, "target_types": _ORGANIZATION + _CREATION + ["FAC"], "priority": 2},
    {"predicate": "acquired", "label": "acquired", "source_types": _ORGANIZATION, "target_types": _ORGANIZATION + _CREATION, "priority": 3},
    {"predicate": "subsidiary_of", "label": "is a subsidiary of", "source_types": _ORGANIZATION, "target_types": _ORGANIZATION, "priority": 3},
    {"predicate": "partnered_with", "label": "is partnered with", "source_types": _ORGANIZATION, "target_types": _ORGANIZATION, "symmetric": True, "priority": 2},
    {"predicate": "competes_with", "label": "competes with", "source_types": _ORGANIZATION, "target_types": _ORGANIZATION, "symmetric": True, "priority": 2},
    {"predicate": "located_in", "label": "is located in", "source_types": _ORGANIZATION + ["FAC"] + _CREATION, "target_types": _PLACE, "priority": 1},
    {"predicate": "headquartered_in", "label": "is headquartered in", "source_types": _ORGANIZATION, "target_types": _PLACE, "priority": 3},
    {"predicate": "part_of", "label": "is part of", "source_types": _ORGANIZATION + _PLACE + ["FAC"] + _CREATION, "target_types": _ORGANIZATION + _PLACE + ["FAC"] + _CREATION, "priority": 1},
    {"predicate": "created", "label": "created", "source_types": _PERSON + _ORGANIZATION, "target_types": _CREATION, "priority": 2},
    {"predicate": "uses", "label": "uses", "source_types": _PERSON + _ORGANIZATION, "target_types": _CREATION, "priority": 1},
    {"predicate": "married_to", "label": "is married to", "source_types": _PERSON, "target_types": _PERSON, "symmetric": True, "priority": 2},
    {"predicate": "parent_of", "label": "is a parent of", "source_types": _PERSON, "target_types": _PERSON, "priority": 3},
    {"predicate": "born_in", "label": "was born in", "source_types": _PERSON, "target_types": _PLACE, "priority": 3},
    {"predicate": "died_in", "label": "died in", "source_types": _PERSON, "target_types": _PLACE, "priority": 3},
    {"predicate": "studied_at", "label": "studied at", "source_types": _PERSON, "target_types": _ORGANIZATION + ["FAC"], "priority": 3},
    {"predicate": "governs", "label": "governs", "source_types": _PERSON + _ORGANIZATION, "target_types": _ORGANIZATION + _PLACE, "priority": 2},
]
_relationship_pipelines: Dict[tuple, Any] = {}
_relationship_pipeline_lock = Lock()
_relationship_inference_lock = Lock()


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


def _pipeline_device(config: Dict[str, Any]) -> int | str:
    configured = str(config.get("device") or "auto").strip().lower()
    if configured != "auto":
        return configured
    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1


def _get_relationship_pipeline(config: Dict[str, Any]):
    model_name = str(
        config.get("model") or "joeddav/xlm-roberta-large-xnli"
    )
    revision = config.get("model_revision")
    device = _pipeline_device(config)
    key = (model_name, revision, device)
    if key not in _relationship_pipelines:
        with _relationship_pipeline_lock:
            if key not in _relationship_pipelines:
                from transformers import pipeline

                _relationship_pipelines[key] = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    revision=revision,
                    device=device,
                )
    return _relationship_pipelines[key]


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
    if len(spans) != 2:
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
    context = content[start:end].strip()
    if len(context) > 500:
        return ""
    return context


def _relation_definitions(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    configured = config.get("relations", _DEFAULT_RELATIONS)
    if not isinstance(configured, list) or not configured:
        raise ValueError("relationship-extraction-map relations must be a non-empty list")
    definitions = []
    defaults = {item["predicate"]: item for item in _DEFAULT_RELATIONS}
    for item in configured:
        if not isinstance(item, dict):
            raise ValueError("relationship-extraction-map relations must be objects")
        predicate = _normalize_predicate(str(item.get("predicate") or ""))
        label = str(item.get("label") or "").strip()
        if not predicate or not label:
            raise ValueError(
                "relationship-extraction-map relations require predicate and label"
            )
        default = defaults.get(predicate, {})
        definitions.append(
            {
                "predicate": predicate,
                "label": label,
                "symmetric": bool(item.get("symmetric", False)),
                "source_types": set(item.get("source_types", default.get("source_types", []))),
                "target_types": set(item.get("target_types", default.get("target_types", []))),
                "priority": int(item.get("priority", default.get("priority", 1))),
            }
        )
    return definitions


def _entity_type(entity: Dict[str, Any]) -> str:
    return str(entity.get("type") or "").strip().upper()


def _supports_types(
    definition: Dict[str, Any],
    subject: Dict[str, Any],
    obj: Dict[str, Any],
) -> bool:
    subject_type = _entity_type(subject)
    object_type = _entity_type(obj)
    return (
        (not subject_type or not definition["source_types"] or subject_type in definition["source_types"])
        and (not object_type or not definition["target_types"] or object_type in definition["target_types"])
    )


def _logit(score: float) -> float:
    bounded = min(max(score, 1e-7), 1 - 1e-7)
    return log(bounded / (1 - bounded))


def _classify_direction(
    classifier: Any,
    context: str,
    subject: Dict[str, Any],
    obj: Dict[str, Any],
    definitions: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates = [
        definition
        for definition in definitions
        if _supports_types(definition, subject, obj)
    ]
    if not candidates:
        return []
    by_label = {definition["label"]: definition for definition in candidates}
    result = classifier(
        [context, f"{subject['name']} and {obj['name']}."],
        candidate_labels=list(by_label),
        hypothesis_template=f"{subject['name']} {{}} {obj['name']}.",
        multi_label=True,
        batch_size=2,
    )
    if not isinstance(result, list) or len(result) != 2:
        return []
    actual = dict(zip(result[0].get("labels", []), result[0].get("scores", [])))
    neutral = dict(zip(result[1].get("labels", []), result[1].get("scores", [])))
    threshold = float(config.get("score_threshold", 0.8))
    evidence_margin = float(config.get("evidence_margin", 2.5))
    specificity_margin = float(config.get("specificity_margin", 0.7))
    scored = []
    for label, definition in by_label.items():
        score = float(actual.get(label, 0))
        evidence = _logit(score) - _logit(float(neutral.get(label, 0.5)))
        if score >= threshold and evidence >= evidence_margin:
            scored.append((definition, score, _logit(score), evidence))
    if not scored:
        return []
    best_logit = max(item[2] for item in scored)
    eligible = [item for item in scored if best_logit - item[2] <= specificity_margin]
    eligible.sort(key=lambda item: (item[0]["priority"], item[2]), reverse=True)
    max_relations = int(config.get("max_relations_per_pair", 1))
    return [
        {
            "subject": subject["name"],
            "predicate": definition["predicate"],
            "object": obj["name"],
            "confidence": 1 / (1 + exp(-evidence)),
            "context": context,
            "symmetric": definition["symmetric"],
            "_evidence": evidence,
        }
        for definition, _score, _actual_logit, evidence in eligible[:max_relations]
    ]


def _classify_relationships(
    content: str,
    entities: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    definitions = _relation_definitions(config)
    threshold = float(config.get("score_threshold", 0.8))
    max_pairs = int(config.get("max_pairs", 64))
    max_relations_per_pair = int(config.get("max_relations_per_pair", 1))
    evidence_margin = float(config.get("evidence_margin", 2.5))
    specificity_margin = float(config.get("specificity_margin", 0.7))
    if not 0 <= threshold <= 1:
        raise ValueError("relationship-extraction-map score threshold is invalid")
    if (
        max_pairs < 1
        or max_relations_per_pair < 1
        or evidence_margin < 0
        or specificity_margin < 0
    ):
        raise ValueError("relationship-extraction-map pair limits are invalid")

    present = [
        entity
        for entity in entities
        if str(entity["name"]).casefold() in content.casefold()
    ]
    pairs = islice(combinations(present, 2), max_pairs)
    classifier = _get_relationship_pipeline(config)
    relationships = []
    with _relationship_inference_lock:
        for first, second in pairs:
            context = _context(content, first["name"], second["name"])
            if not context:
                continue
            pair_results = _classify_direction(
                classifier, context, first, second, definitions, config
            ) + _classify_direction(
                classifier, context, second, first, definitions, config
            )
            pair_results.sort(key=lambda item: item["_evidence"], reverse=True)
            for relationship in pair_results[:max_relations_per_pair]:
                relationship.pop("_evidence", None)
                relationships.append(relationship)
    return relationships


def _deduplicate_classified(
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    order = []
    for relationship in relationships:
        endpoints = (relationship["subject"], relationship["object"])
        if relationship.pop("symmetric", False):
            endpoints = tuple(sorted(endpoints, key=str.casefold))
            relationship["subject"], relationship["object"] = endpoints
        key = (endpoints[0], relationship["predicate"], endpoints[1])
        if key not in best:
            best[key] = relationship
            order.append(key)
        elif relationship["confidence"] > best[key]["confidence"]:
            best[key] = relationship
    return [best[key] for key in order]


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
    return {
        "relationships": _deduplicate_classified(
            _classify_relationships(safe_content, entities, config)
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
