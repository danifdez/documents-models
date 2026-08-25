"""Self-contained steps for the durable entity-extraction workflow."""

import json
import re
from typing import Any, Dict, List

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs
from services.llm_service import get_llm_service

_PROMPT = get_prompt("entity-extraction-map")
_ALLOWED_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "EVENT",
    "FAC",
    "PRODUCT",
    "WORK_OF_ART",
    "LANGUAGE",
    "LAW",
}
_PROPER_NAME_LABELS = _ALLOWED_LABELS - {"NORP", "LANGUAGE"}


def _looks_like_proper_name(word: str) -> bool:
    return any(character.isupper() for character in word)


def _extract_entities(
    content: str,
    config: Dict[str, Any],
) -> List[Dict[str, str]]:
    safe_content = strip_dense_blobs(content).strip()
    if not safe_content:
        raise ValueError("entity-extraction-map requires non-empty content")
    max_input_words = int(config.get("max_input_words", 1500))
    if len(safe_content.split()) > max_input_words:
        raise ValueError("entity-extraction-map content exceeds its word budget")
    if not _PROMPT:
        raise RuntimeError("entity-extraction-map prompt is unavailable")

    response = get_llm_service(
        **get_llm_params("entity-extraction-map")
    ).chat(
        [{"role": "user", "content": _PROMPT.format(text=safe_content)}],
        max_tokens=int(config.get("max_tokens", 2000)),
        temperature=0.0,
    )
    try:
        parsed = json.loads(
            re.sub(
                r"^\s*```(?:json)?\s*|\s*```\s*$",
                "",
                response.strip(),
            )
        )
    except (AttributeError, json.JSONDecodeError) as error:
        raise ValueError("entity-extraction-map returned invalid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError("entity-extraction-map result must be an array")

    ignored = set(config.get("ignored_entity_types", []))
    result: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word") or "").strip()
        entity = str(item.get("entity") or "").strip().upper()
        if len(word) <= 1 or entity not in _ALLOWED_LABELS or entity in ignored:
            continue
        if entity in _PROPER_NAME_LABELS and not _looks_like_proper_name(word):
            continue
        result.append({"word": word, "entity": entity})
    return result


def _dedupe(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    result: List[Dict[str, str]] = []
    for entry in entities:
        key = entry["word"].casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


@execution_handler("entity-extraction-map")
def entity_extraction_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("entity-extraction-map content must be a string")
    return {
        "entities": _extract_entities(
            content,
            get_task_config("entity-extraction-map"),
        )
    }


@execution_handler("entity-extraction-reduce")
def entity_extraction_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("entity-extraction-reduce requires partials")

    entities: List[Dict[str, str]] = []
    for partial in partials:
        if not isinstance(partial, list):
            raise ValueError("entity-extraction-reduce partials must be arrays")
        for entry in partial:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("word"), str)
                or not entry["word"].strip()
                or entry.get("entity") not in _ALLOWED_LABELS
            ):
                raise ValueError("entity-extraction-reduce received an invalid entity")
            entities.append(
                {"word": entry["word"].strip(), "entity": entry["entity"]}
            )
    return {"entities": _dedupe(entities)}
