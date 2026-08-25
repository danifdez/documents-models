"""Entity-extraction task.

Runs through the shared inline map-reduce helper (`lib.llm.map_reduce`):

- The root invocation cleans the text, splits it into section units and chunks
  them.
- Each chunk returns its entities and the reduce step concatenates and
  deduplicates them.

Chunking is not an optimisation here: the previous single-shot version pushed
the whole document through `truncate_for_llm`, so everything past the model's
budget was dropped without a trace. A long document silently lost the entities
of its tail.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from lib.llm.config import get_llm_params, get_task_config
from lib.llm.map_reduce import MapReduceSpec, run_map_reduce
from lib.llm.prompts import get_prompt
from lib.llm.text import build_chunks, strip_dense_blobs, truncate_for_llm
from services.llm_service import get_llm_service
from common.execution_registry import execution_handler

logger = logging.getLogger(__name__)

_PROMPT = get_prompt("entity-extraction")
_ALLOWED_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "FAC", "PRODUCT",
    "WORK_OF_ART", "LANGUAGE", "LAW",
}

# A named entity is, by definition, a proper name. The prompt already forbids
# roles and generic nouns, but an 8B model still answers `coordinadora` or
# `centro cívico`, so the guarantee has to live here rather than in the wording.
# NORP and LANGUAGE are exempt: "españoles", "français" and most language names
# are lowercase outside English, and demanding a capital would drop them.
_PROPER_NAME_LABELS = _ALLOWED_LABELS - {"NORP", "LANGUAGE"}


def _looks_like_proper_name(word: str) -> bool:
    """Any uppercase letter anywhere is enough.

    Deliberately permissive: German capitalises every noun, so a stricter rule
    would not discriminate there anyway, and the cost of a false negative (a
    real entity dropped) is higher than that of a false positive.
    """
    return any(ch.isupper() for ch in word)


def _payload_text(payload: Dict[str, Any]) -> str:
    """Flatten `texts` (list of strings or {text: ...}) / `text` into one blob."""
    texts = payload.get("texts") or ([payload["text"]] if payload.get("text") else [])
    parts: List[str] = []
    for item in texts:
        if isinstance(item, dict) and "text" in item:
            parts.append(strip_dense_blobs(str(item["text"])))
        else:
            parts.append(strip_dense_blobs(str(item)))
    return "\n\n".join(parts)


def _extract_entities(text: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Entities of a SINGLE chunk. Cross-chunk dedupe belongs to the merge."""
    safe_text = truncate_for_llm(strip_dense_blobs(text), config)
    if not safe_text.strip() or not _PROMPT:
        return []
    try:
        # No grammar on purpose: constraining this call makes the model answer
        # a bare `[]` in en/de/fr (it emits `[` as its own token and greedy
        # decoding then picks the closing bracket). The fence-stripping parse
        # below is what actually works with this model.
        response = get_llm_service(**get_llm_params("entity-extraction")).chat(
            [{"role": "user", "content": _PROMPT.format(text=safe_text)}],
            max_tokens=int(config.get("max_tokens", 2000)), temperature=0.0,
        )
        parsed = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response.strip()))
    except Exception:
        logger.exception("entity-extraction chat failed")
        return []

    ignored = set(config.get("ignored_entity_types", []))
    result: List[Dict[str, str]] = []
    for item in parsed if isinstance(parsed, list) else []:
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
    """One row per entity, in first-appearance order.

    Keyed on the word alone, not on (word, label): the same name classified as
    PERSON in one chunk and ORG in the next is one entity the model labelled
    inconsistently, and listing it twice would be wrong.
    """
    seen = set()
    out: List[Dict[str, str]] = []
    for entry in entities:
        key = entry["word"].casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _chunks(payload: Dict[str, Any], cfg: Dict[str, Any], is_child: bool) -> List[str]:
    raw_content = _payload_text(payload)
    if not raw_content.strip():
        return []
    return build_chunks(raw_content, int(cfg.get("chunk_word_budget", 1500)))


def _leaf(chunk: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    return _extract_entities(chunk, cfg)


def _reduce(partials: List[Any], payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for entries_list in partials:
        for entry in (entries_list or []):
            if isinstance(entry, dict) and entry.get("word") and entry.get("entity"):
                merged.append({"word": str(entry["word"]), "entity": str(entry["entity"])})
    return _dedupe(merged)


_SPEC = MapReduceSpec(
    leaf_fn=_leaf,
    reduce_fn=_reduce,
    chunk_field="text",
    recursive_merge=False,
    result_key="entities",
    empty_value=[],
    list_results=True,
    chunks_fn=_chunks,
)


@execution_handler("entity-extraction")
def entities(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, Any]:
    """Extract named entities from text using the local LLM.

    Payload:
        texts: list of strings or {text: string} dicts — or `text`, a single one.

    Returns:
        {"entities": [{"word": str, "entity": str}, ...]} or {"error": str}.
    """
    try:
        cfg = get_task_config("entity-extraction")
        return run_map_reduce(payload, state, ctx, spec=_SPEC, cfg=cfg)
    except Exception as e:
        logger.exception("entity-extraction failed")
        return {"error": f"entity-extraction failed: {e}"}
