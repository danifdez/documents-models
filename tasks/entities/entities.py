"""Reentrant entity-extraction task.

Same state-machine shape as `keywords` and `date-extraction`:

- The root invocation cleans the text, splits it into section units and chunks
  them. One chunk runs inline; several fan out one child job per chunk.
- Each child returns the entities of its own chunk.
- Once every child is done the dispatcher re-invokes the handler with the
  persisted state and `_phase_merge` concatenates and dedupes.

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
from lib.llm.prompts import get_prompt
from lib.llm.text import build_chunks, strip_dense_blobs, truncate_for_llm
from services.llm_service import get_llm_service
from utils.job_registry import job_handler

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


def _phase_plan_or_leaf(payload: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    is_child = "_chunk_idx" in payload
    raw_content = _payload_text(payload)
    if not raw_content.strip():
        return {"entities": []}

    chunks = build_chunks(raw_content, int(cfg.get("chunk_word_budget", 1500)))
    if not chunks:
        return {"entities": []}

    # CHILD: raw per-chunk entities; the parent dedupes across chunks.
    if is_child:
        found: List[Dict[str, str]] = []
        for c in chunks:
            found.extend(_extract_entities(c, cfg))
        return {"entities": found}

    if len(chunks) == 1:
        return {"entities": _dedupe(_extract_entities(chunks[0], cfg))}

    # No job queue (unit tests / fallback): in-process serial.
    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        found = []
        for c in chunks:
            found.extend(_extract_entities(c, cfg))
        return {"entities": _dedupe(found)}

    # FAN-OUT: one child per chunk.
    pending: Dict[str, int] = {}
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    retries: Dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        child_id = ctx.db.enqueue_child_job(
            ctx.job_id,
            "entity-extraction",
            payload={"text": chunk, "_chunk_idx": i},
            agent_max_steps=1,
        )
        if child_id is None:
            return {"error": f"failed to enqueue child for chunk {i}"}
        pending[str(child_id)] = i
        results[str(i)] = None
        retries[str(i)] = 0

    state = {
        "phase": "merging",
        "chunks_count": len(chunks),
        "pending": pending,
        "results": results,
        "retries": retries,
        "chunks": chunks,
        "chunk_field": "text",
        "chunk_payload_template": {},
    }
    return {
        "_sub_agent_pending_many": True,
        "_state": state,
        "pending_children": pending,
    }


def _phase_merge(state: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    failed_idx = state.get("failed_idx")
    if failed_idx is not None:
        return {
            "error": (
                f"chunk {failed_idx} failed after retries: "
                f"{state.get('failed_error') or 'unknown error'}"
            )
        }

    results = state.get("results") or {}
    merged: List[Dict[str, str]] = []
    for i in range(int(state.get("chunks_count", 0))):
        chunk_result = results.get(str(i))
        if not isinstance(chunk_result, dict):
            continue
        for entry in chunk_result.get("entities") or []:
            if isinstance(entry, dict) and entry.get("word") and entry.get("entity"):
                merged.append({"word": str(entry["word"]), "entity": str(entry["entity"])})

    return {"entities": _dedupe(merged)}


@job_handler("entity-extraction")
def entities(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, Any]:
    """Extract named entities from text using the local LLM.

    Reentrant: `state` is None on the first invocation and carries
    `phase == "merging"` when the dispatcher wakes the parent after its
    children finish.

    Payload:
        texts: list of strings or {text: string} dicts — or `text`, a single one.

    Returns:
        {"entities": [{"word": str, "entity": str}, ...]} or {"error": str}.
    """
    try:
        cfg = get_task_config("entity-extraction")
        if state and state.get("phase") == "merging":
            return _phase_merge(state, cfg, ctx)
        return _phase_plan_or_leaf(payload, cfg, ctx)
    except Exception as e:
        logger.exception("entity-extraction failed")
        return {"error": f"entity-extraction failed: {e}"}
